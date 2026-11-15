import torch
import numpy as np
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from tensordict import TensorDict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.tracking import Tracking
import shutil
import requests
from timeagent.search.client import RetrievalClient, RetrievalClientError

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3
    reward_type: str = 'continuous'
    search_timeout: float = 30.0
    search_token_env: str = None
    search_max_response_bytes: int = 20_000_000

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        # Validate transport configuration before a rollout can increment the
        # executed-query counters. Tokens stay in the environment/client only.
        self.search_client = (RetrievalClient(
            config.search_url, timeout=config.search_timeout, topk=config.topk,
            token_env=config.search_token_env,
            max_response_bytes=config.search_max_response_bytes)
            if config.search_url else None)

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [resp.split('</search>')[0] + '</search>'
                 if '</search>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """Pad active/tail batches for dispatch and preserve validation metadata."""
        divisor = getattr(self.actor_rollout_wg, 'world_size', self.config.num_gpus)
        for key in active_batch.batch.keys():
            value = active_batch.batch[key]
            if not isinstance(value, TensorDict):
                active_batch.batch[key] = value.long()
        padded, pad_size = pad_dataproto_to_divisor(active_batch, divisor)
        output = self.actor_rollout_wg.generate_sequences(padded)
        return unpad_dataproto(output, pad_size)

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop with executor-level query counts."""
        lengths = gen_batch.batch['attention_mask'].sum(dim=1)
        if (lengths > self.config.max_start_length).any():
            raise ValueError('Input exceeds max_start_length; increase it instead of silently truncating documents')
        self._executed_searches = [0] * len(gen_batch)
        self._failed_searches = [0] * len(gen_batch)
        meta_info = dict(gen_batch.meta_info)
        if self.config.reward_type == 'naiverag':
            return self._run_naiverag_loop(gen_batch, initial_input_ids)

        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}

        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            }, meta_info=dict(rollings.meta_info))            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask
            )
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            
        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            }, meta_info=dict(rollings.meta_info))            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            _, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = self._executed_searches.copy()
        meta_info['executed_searches'] = self._executed_searches.copy()
        meta_info['failed_searches'] = self._failed_searches.copy()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _extract_original_query(self, text: str) -> str:
        marker = 'Content:'
        if marker not in text:
            return text
        return text.split(marker, 1)[1].strip()

    def _run_naiverag_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Naive RAG: retrieve once using Content: query, then generate answer directly."""
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}

        prompts = self.tokenizer.batch_decode(original_left_side['input_ids'], skip_special_tokens=True)
        search_queries = [self._extract_original_query(p) for p in prompts]
        search_results = self._execute_search_queries(list(range(len(search_queries))), search_queries)
        next_obs = [f'\nThe relevant search information is stored in <information>:\n<information>{r.strip()}</information>\n' for r in search_results]
        next_obs_ids = self._process_next_obs(next_obs)

        rollings = self._update_rolling_state(gen_batch, initial_input_ids[:, []], next_obs_ids)
        original_right_side = self._update_right_side(original_right_side, initial_input_ids[:, []], next_obs_ids)

        rollings.batch = self.tensor_fn.cut_to_effective_len(rollings.batch, keys=['input_ids', 'attention_mask', 'position_ids'])
        gen_output = self._generate_with_gpu_padding(rollings)
        meta_info = gen_output.meta_info

        responses_str = self.tokenizer.batch_decode(gen_output.batch['responses'], skip_special_tokens=True)
        responses_str = [r.split('</answer>')[0] + '</answer>' if '</answer>' in r else r for r in responses_str]
        responses_ids = self._batch_tokenize(responses_str)

        original_right_side = self._update_right_side(original_right_side, responses_ids)

        batch_size = initial_input_ids.shape[0]
        meta_info['turns_stats'] = [1] * batch_size
        meta_info['active_mask'] = [False] * batch_size
        meta_info['valid_action_stats'] = [1] * batch_size
        meta_info['valid_search_stats'] = self._executed_searches.copy()
        meta_info['executed_searches'] = self._executed_searches.copy()
        meta_info['failed_searches'] = self._failed_searches.copy()
        print("ACTIVE_TRAJ_NUM:", [batch_size, 0])

        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        counters = {key: np.array(meta_info[key], dtype=object)
                    for key in ('executed_searches', 'failed_searches')}
        final_output = DataProto.from_dict(final_output, non_tensors=counters)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def _execute_search_queries(self, indexes, queries):
        """Count dispatched per-document queries, including failed HTTP attempts.

        One batched HTTP request containing k queries counts as one search for
        each of k documents. Empty/inactive/budget-rejected actions count zero.
        Failed requests return an explicit failure observation and are logged.
        """
        if not queries:
            return []
        for index in indexes:
            self._executed_searches[index] += 1
        try:
            results = self.batch_search(queries)
            if len(results) != len(queries):
                raise ValueError('Search result count does not match dispatched query count')
            return results
        except (requests.RequestException, ValueError, KeyError, TypeError):
            for index in indexes:
                self._failed_searches[index] += 1
            return ['Search request failed; no external evidence was returned.'] * len(queries)

    def execute_predictions(self, predictions, pad_token, active_mask=None, do_search=True):
        actions, contents = self.postprocess_predictions(predictions)
        if active_mask is None:
            active_mask = [True] * len(predictions)
        if len(active_mask) != len(predictions):
            raise ValueError('Active mask and predictions must align')
        indexes = [i for i, (action, active) in enumerate(zip(actions, active_mask))
                   if active and action == 'search' and contents[i] and do_search]
        results = self._execute_search_queries(indexes, [contents[i] for i in indexes])
        result_by_index = dict(zip(indexes, results))
        observations, dones, valid, searched = [], [], [], []
        for i, (action, active) in enumerate(zip(actions, active_mask)):
            if not active or action == 'answer':
                observations.append(''); dones.append(1)
                valid.append(int(bool(active) and action == 'answer')); searched.append(0)
            elif i in result_by_index:
                observations.append('\n\n<information>' + result_by_index[i].strip() + '</information>\n\n')
                dones.append(0); valid.append(1); searched.append(1)
            elif action == 'search' and not do_search:
                # A search string after the budget is exhausted is not execution.
                observations.append(''); dones.append(1); valid.append(0); searched.append(0)
            else:
                observations.append('\nInvalid action. Emit a nonempty <search>query</search> or <answer>YYYY-MM-DD</answer>.\n')
                dones.append(0); valid.append(0); searched.append(0)
        return observations, dones, valid, searched

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        if not queries:
            return []
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        if self.search_client is None:
            raise ValueError('No retrieval service URL was configured')
        try:
            return self.search_client.retrieve(queries, topk=self.config.topk)
        except RetrievalClientError as error:
            # The executor handles this as one failed attempt per document.
            # Never retry: a timed-out request may already have executed.
            raise ValueError('Retrieval failed: ' + error.kind) from None

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            publish_time = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Publish time: {publish_time}) {text}\n"

        return format_reference
