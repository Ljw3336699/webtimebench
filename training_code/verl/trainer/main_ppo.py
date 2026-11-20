"""TIME-Agent PPO entry: cost-aware date reward with real executed-search accounting."""
import os
import re
import hydra
import torch
import ray
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.single_controller.base import Worker
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.workers.fsdp_workers import RewardModelWorker
from verl.utils.reward_score import qa_em


def _select_rm_score_fn(data_source):
    return qa_em.compute_score_em


class RewardManager:
    """Date reward with explicit training/evaluation separation."""

    def __init__(self, tokenizer, num_examine, use_search_bonus=True,
                 reward_type='cost_aware', reward_tag='full', eval_mode=False,
                 cost_aware=None, reward_diagnostics_path=None):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.use_search_bonus = use_search_bonus
        self.reward_type = reward_type
        self.reward_tag = reward_tag
        self.eval_mode = eval_mode
        self.cost_aware = cost_aware
        self.reward_diagnostics_path = reward_diagnostics_path
        self.reward_batch_index = 0
        if reward_type == 'cost_aware' and not eval_mode:
            from verl.utils.reward_score.reward_objectives import CostAwareConfig
            self.cost_aware = CostAwareConfig.from_mapping(cost_aware).as_dict()

    def __call__(self, data: DataProto, return_diagnostics=False):
        from verl.utils.reward_score.temporal import reference_date, normalize_regime, validate_search_count
        if 'rm_scores' in data.batch:
            if self.eval_mode or return_diagnostics or self.reward_type == 'cost_aware':
                raise ValueError('Configured date scoring cannot bypass parsing/cost with rm_scores')
            return data.batch['rm_scores']
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        diagnostics = {key: [] for key in ('predictions', 'targets', 'difficulties',
                      'reasoning_paths', 'prompt_tokens', 'response_tokens', 'ids',
                      'executed_searches', 'failed_searches', 'reward_components')}
        components = []
        for i in range(len(data)):
            item = data[i]
            prompt_ids = item.batch['prompts']
            prompt_width = prompt_ids.shape[-1]
            prompt_length = int(item.batch['attention_mask'][:prompt_width].sum())
            response_length = int(item.batch['attention_mask'][prompt_width:].sum())
            valid_prompt_ids = prompt_ids[-prompt_length:] if prompt_length else prompt_ids[:0]
            response_ids = item.batch['responses'][:response_length]
            # Responses contain retrieved observations in agent rollouts. Mask them
            # out before parsing so a retrieved <answer> cannot become a prediction.
            model_ids = response_ids
            if 'info_mask' in item.batch:
                model_mask = item.batch['info_mask'][prompt_width:prompt_width + response_length].bool()
                model_ids = response_ids[model_mask]
            if not self.eval_mode and len(model_ids) == 0:
                # An empty rollout has no action to receive its terminal reward.
                # Silently leaving zeros would erase invalid-answer/call costs.
                # Opt-in runtime switch: score it as an invalid answer instead of
                # aborting a long run; the invalid penalty and search cost still apply.
                if os.environ.get('TIMEAGENT_ALLOW_EMPTY_ROLLOUT') != '1':
                    raise ValueError('Training rollout has no model-generated response tokens')
            model_ids = model_ids.long() if hasattr(model_ids, 'long') else model_ids
            joined = torch.cat((valid_prompt_ids, response_ids))
            joined = joined.long() if hasattr(joined, 'long') else joined
            model_text = self.tokenizer.decode(model_ids, skip_special_tokens=True)
            full_text = self.tokenizer.decode(joined, skip_special_tokens=True)
            raw_gold = item.non_tensor_batch['reward_model']['ground_truth']
            target = reference_date(raw_gold)
            difficulty = normalize_regime(item.non_tensor_batch.get('difficulty'))
            ground_truth = {'target': [target], 'difficulty': difficulty}
            count = item.non_tensor_batch.get('executed_searches')
            failures = item.non_tensor_batch.get('failed_searches')
            count = validate_search_count(count)
            failures = validate_search_count(failures)
            if failures is not None and count is not None and failures > count:
                raise ValueError('failed_searches cannot exceed executed_searches')
            details = _select_rm_score_fn(item.non_tensor_batch['data_source'])(
                solution_str=model_text, ground_truth=ground_truth,
                use_search_bonus=self.use_search_bonus, reward_type=self.reward_type,
                reward_tag=self.reward_tag, eval_mode=self.eval_mode,
                executed_searches=count, cost_aware=self.cost_aware, return_details=True)
            score = details['total_reward']
            components.append({'id': item.non_tensor_batch.get('id'),
                               'reward_batch_index': self.reward_batch_index,
                               'failed_searches': failures, **details})
            if response_length:
                reward_tensor[i, response_length - 1] = score
            if i < self.num_examine:
                print(model_text)
            if return_diagnostics:
                diagnostics['predictions'].append(details['prediction'])
                diagnostics['targets'].append(target)
                diagnostics['difficulties'].append(difficulty)
                diagnostics['reasoning_paths'].append(full_text)
                diagnostics['prompt_tokens'].append(prompt_length)
                diagnostics['response_tokens'].append(len(model_ids))
                diagnostics['ids'].append(item.non_tensor_batch.get('id'))
                diagnostics['executed_searches'].append(count)
                diagnostics['failed_searches'].append(failures)
                diagnostics['reward_components'].append(components[-1])
        if not self.eval_mode:
            # IDs keep these records interpretable after training batches reorder.
            if hasattr(data, 'meta_info'):
                data.meta_info['reward_diagnostics'] = components
            if self.reward_diagnostics_path:
                import json
                from pathlib import Path
                path = Path(self.reward_diagnostics_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('a', encoding='utf-8') as stream:
                    for row in components:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            self.reward_batch_index += 1
        if return_diagnostics:
            return {'reward_tensor': reward_tensor, **diagnostics}
        return reward_tensor




@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    import os
    seed = int(os.environ.get('TIMEAGENT_SEED', config.get('seed', 20261103)))
    if seed < 0:
        raise ValueError('Seed must be nonnegative')
    with open_dict(config):
        config.seed = seed
        config.actor_rollout_ref.seed = seed
        config.actor_rollout_ref.rollout.seed = seed
        config.critic.seed = seed
    retrieval_env = {}
    token_name = config.retriever.get('token_env')
    if token_name:
        if not isinstance(token_name, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', token_name):
            raise ValueError('retriever.token_env must name an environment variable')
        token = os.environ.get(token_name)
        if not token or any(ord(ch) < 33 or ord(ch) > 126 for ch in token):
            raise ValueError('Retrieval token environment variable is missing, empty or malformed')
        retrieval_env[token_name] = token
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN', 'TIMEAGENT_SEED': str(seed)}})

    # A pre-existing Ray cluster may not inherit the current shell's token.
    # Only the explicitly selected variable is passed, outside printed config.
    task = main_task.options(runtime_env={'env_vars': retrieval_env}) if retrieval_env else main_task
    ray.get(task.remote(config))


@ray.remote
def main_task(config):
    from transformers import set_seed
    set_seed(int(config.seed))
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.utils import hf_tokenizer
    import omegaconf
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup
    else:
        raise NotImplementedError

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }
    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_type = config.reward_model.get('reward_type', 'cost_aware')
    reward_tag = config.reward_model.get('reward_tag', 'full')

    # Training reward function: reward_tag controls ablation variant.
    cost_aware_options = config.reward_model.get('cost_aware')
    if cost_aware_options is not None:
        cost_aware_options = OmegaConf.to_container(cost_aware_options, resolve=True)
    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, use_search_bonus=True,
                             reward_type=reward_type, reward_tag=reward_tag,
                             cost_aware=cost_aware_options,
                             reward_diagnostics_path=config.reward_model.get('reward_diagnostics_path'))

    # Validation always reports unweighted date utility, independent of training reward.
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, use_search_bonus=False, reward_type='continuous', eval_mode=True)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    try:
        trainer.fit()
    finally:
        trainer._store_ray_worker_pool_alive_status = lambda: None
        del trainer


if __name__ == '__main__':
    main()