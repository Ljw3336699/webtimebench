#!/usr/bin/env python3
"""Build or run the corrected PPO/evaluation command; training validates on dev."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["train", "evaluate"])
    p.add_argument("--model", required=True, help="Local base-model/checkpoint path or model identifier")
    p.add_argument("--variant", choices=["clean", "temporal"], default="clean")
    p.add_argument("--dataset-dir", type=Path, help="Custom web dataset root containing manifest.json and train/dev/test.parquet")
    p.add_argument("--eval-split", choices=["dev", "test"], default="test", help="Only used in evaluate mode")
    p.add_argument("--reward", choices=["cost_aware", "continuous", "timeagent", "searchr1"], default="cost_aware")
    p.add_argument("--quality-scales-days", type=float, nargs="+", default=[7, 30, 365])
    p.add_argument("--quality-weights", type=float, nargs="+", default=[.2, .3, .5])
    p.add_argument("--search-cost", type=float, default=.01)
    p.add_argument("--cost-by-regime", default=None,
                   help="Per-regime search cost, e.g. 'EXP:0.2,IMP:0.01,DIS:0.01' (overrides search-cost per regime)")
    p.add_argument("--invalid-penalty", type=float, default=.25)
    p.add_argument("--train-batch-size", type=int, default=512)
    p.add_argument("--ppo-mini-batch-size", type=int, default=128)
    p.add_argument("--micro-batch-per-gpu", type=int, default=2)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    p.add_argument("--val-batch-size", type=int, default=256)
    p.add_argument("--no-search", action="store_true")
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=20261103)
    p.add_argument("--search-budget", type=int, choices=range(5), default=2)
    p.add_argument("--retriever-url", default="http://127.0.0.1:8000/retrieve")
    p.add_argument("--retriever-timeout", type=float, default=30)
    p.add_argument("--retriever-token-env", help="Environment variable name; token values never enter launch arguments")
    p.add_argument("--retriever-max-response-bytes", type=int, default=20_000_000)
    p.add_argument("--run-name", help="Optional run directory name; a UTC timestamp is used otherwise")
    p.add_argument("--logger", default="console",
                   help="Comma list of trainer logger backends, e.g. console,tensorboard,wandb")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="offline",
                   help="offline keeps a fully local wandb run dir that can be synced later")
    p.add_argument("--run", action="store_true", help="Execute; without this flag only validate data and print the command")
    p.add_argument("--code-root", type=Path, help="Code tree to run (default: <repo>/src)")
    return p


def parse_cost_by_regime(value):
    """Parse 'EXP:0.2,IMP:0.01,DIS:0.01' into a JSON dict for Hydra."""
    if value is None:
        return "null"
    parsed = {}
    for part in value.split(","):
        part = part.strip()
        if ":" not in part:
            raise ValueError("cost-by-regime requires REGIME:COST pairs, e.g. EXP:0.2,IMP:0.01,DIS:0.01")
        regime, raw = part.split(":", 1)
        regime = regime.strip().upper()
        cost = float(raw.strip())
        if regime not in ("EXP", "IMP", "DIS") or cost < 0:
            raise ValueError(f"Invalid cost-by-regime entry: {part}")
        if regime in parsed:
            raise ValueError(f"Duplicate regime in cost-by-regime: {regime}")
        parsed[regime] = cost
    if len(parsed) < 2:
        raise ValueError("cost-by-regime should cover at least EXP and IMP/DIS")
    # Hydra override grammar accepts unquoted dict keys: {EXP:0.2,IMP:0.01,DIS:0.01}
    return "{" + ",".join(f"{k}:{v}" for k, v in parsed.items()) + "}"


def build_command(a, root=ROOT):
    if (not math.isfinite(a.retriever_timeout) or a.retriever_timeout <= 0
            or a.retriever_max_response_bytes < 1):
        raise ValueError("Retriever timeout and maximum response bytes must be positive")
    if a.retriever_token_env is not None and not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', a.retriever_token_env):
        raise ValueError("retriever-token-env must name an environment variable")
    if any(x < 1 for x in (a.gpus, a.steps, a.train_batch_size, a.ppo_mini_batch_size,
                           a.micro_batch_per_gpu, a.val_batch_size)) or a.seed < 0:
        raise ValueError("Batch sizes, gpus and steps must be positive; seed must be nonnegative")
    if not 0 < a.gpu_memory_utilization <= 1:
        raise ValueError("gpu-memory-utilization must be in (0, 1]")
    global_micro = a.gpus * a.micro_batch_per_gpu
    if a.ppo_mini_batch_size % global_micro or a.train_batch_size % a.ppo_mini_batch_size:
        raise ValueError("PPO minibatch must divide train batch and be divisible by GPUs * per-GPU microbatch")
    if (len(a.quality_scales_days) != len(a.quality_weights) or not a.quality_weights
            or any(not math.isfinite(x) or x <= 0 for x in a.quality_scales_days)
            or any(not math.isfinite(x) or x < 0 for x in a.quality_weights)
            or not math.isclose(sum(a.quality_weights), 1, abs_tol=1e-9)
            or any(not math.isfinite(x) or x < 0 for x in (a.search_cost, a.invalid_penalty))):
        raise ValueError("Invalid cost-aware reward configuration")
    name = a.run_name or (a.mode + "_" + a.variant + "_" + a.reward + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("run-name must be a single directory name")
    out = root / "outputs" / name
    data = a.dataset_dir.resolve() if a.dataset_dir else root / "data" / a.variant
    val_split = "dev" if a.mode == "train" else a.eval_split
    search = not a.no_search and a.search_budget > 0
    # This veRL version divides global microbatches by data-parallel world size.
    # One sample per GPU avoids a zero-sized batch under the default 8-GPU run.
    values = {
        "data.train_files": str(data / "train.parquet"),
        "data.val_files": str(data / f"{val_split}.parquet"),
        "data.train_data_num": "null", "data.val_data_num": "null",
        "data.train_batch_size": a.train_batch_size, "data.val_batch_size": a.val_batch_size,
        "data.max_prompt_length": 16384, "data.max_start_length": 8192,
        "data.max_response_length": 512, "data.max_obs_length": 2048,
        "data.shuffle_train_dataloader": "true",
        "algorithm.adv_estimator": "gae", "algorithm.no_think_rl": "false",
        "algorithm.kl_ctrl.kl_coef": 0.001,
        "actor_rollout_ref.model.path": a.model,
        "actor_rollout_ref.model.enable_gradient_checkpointing": "true",
        "actor_rollout_ref.model.use_remove_padding": "true",
        "actor_rollout_ref.actor.optim.lr": "1e-6",
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio": 0.285,
        "actor_rollout_ref.actor.ppo_mini_batch_size": a.ppo_mini_batch_size,
        "actor_rollout_ref.actor.ppo_micro_batch_size": global_micro,
        "actor_rollout_ref.actor.state_masking": "true",
        "actor_rollout_ref.actor.fsdp_config.param_offload": "true",
        "actor_rollout_ref.actor.fsdp_config.grad_offload": "true",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": "true",
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.gpu_memory_utilization": a.gpu_memory_utilization,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size": 128,
        "actor_rollout_ref.ref.log_prob_micro_batch_size": 128,
        "actor_rollout_ref.rollout.n_agent": 1,
        "actor_rollout_ref.rollout.temperature": 1,
        "critic.model.path": a.model,
        "critic.model.enable_gradient_checkpointing": "true",
        "critic.model.use_remove_padding": "true",
        "critic.optim.lr": "1e-5",
        "critic.optim.lr_warmup_steps_ratio": 0.015,
        "critic.ppo_micro_batch_size": global_micro,
        # Explicit global value: the yaml interpolation ${critic.ppo_micro_batch_size}
        # resolves after that key is already divided by world size (becomes 0).
        "critic.forward_micro_batch_size": global_micro,
        "critic.model.fsdp_config.param_offload": "true",
        "critic.model.fsdp_config.grad_offload": "true",
        "critic.model.fsdp_config.optimizer_offload": "true",
        "reward_model.reward_type": a.reward,
        "reward_model.cost_aware.scales_days": json.dumps(a.quality_scales_days, separators=(",", ":")),
        "reward_model.cost_aware.weights": json.dumps(a.quality_weights, separators=(",", ":")),
        "reward_model.cost_aware.search_cost": a.search_cost,
        "reward_model.cost_aware.invalid_penalty": a.invalid_penalty,
        "reward_model.cost_aware.cost_by_regime": parse_cost_by_regime(a.cost_by_regime),
        "reward_model.reward_diagnostics_path": str(out / "training_reward_components.jsonl"),
        "trainer.logger": "[" + ",".join(x.strip() for x in a.logger.split(",") if x.strip()) + "]",
        "trainer.n_gpus_per_node": a.gpus,
        "trainer.nnodes": 1, "trainer.critic_warmup": 0, "trainer.total_training_steps": a.steps,
        # Even a one-batch dataset can reach the requested update count.
        # fit() stops exactly at total_training_steps.
        "trainer.total_epochs": a.steps, "trainer.save_freq": 25,
        # Less frequent validation: each val pass has low GPU utilization windows
        # (short generations + engine wake/sleep) and repeated passes risk idle-reclaim.
        "trainer.test_freq": 100, "trainer.project_name": "TIME-Agent-v2",
        "trainer.experiment_name": name,
        "trainer.default_hdfs_dir": "null", "trainer.default_local_dir": str(out / "checkpoints"),
        "+trainer.dialog_dir": str(out / "predictions"),
        "+trainer.val_only": str(a.mode == "evaluate").lower(),
        "+trainer.val_before_train": "true",
        "max_turns": a.search_budget, "do_search": str(search).lower(),
        "retriever.url": a.retriever_url, "retriever.topk": 3,
        "retriever.timeout": a.retriever_timeout,
        "retriever.token_env": a.retriever_token_env or "null",
        "retriever.max_response_bytes": a.retriever_max_response_bytes,
    }
    # JSON quoting is Hydra syntax here; subprocess receives argv, never a shell.
    argv = [sys.executable, "-m", "verl.trainer.main_ppo"]
    for k, v in values.items():
        if isinstance(v, str) and (" " in v or k.endswith(".path") or k.endswith("_files")):
            v = json.dumps(v)
        argv.append(f"{k}={v}")
    return argv, out, val_split


def main():
    a = parser().parse_args()
    argv, out, split = build_command(a)
    # Fail before a costly model load if data hashes, IDs, labels or splits changed.
    if a.dataset_dir:
        from build_web_dataset import validate_web_dataset
        data_root = a.dataset_dir.resolve()
        result = validate_web_dataset(data_root)
        manifest_path = data_root / "manifest.json"
    else:
        from build_dataset import validate_dataset
        result = validate_dataset(ROOT / "data")
        if not result["parquet_json_equivalent"]:
            raise SystemExit("Build Parquet data before launching a model")
        data_root = ROOT / "data" / a.variant
        manifest_path = ROOT / "data/manifest.json"
    for needed_split in ("train", split):
        if not (data_root / f"{needed_split}.parquet").is_file():
            raise SystemExit(f"Missing {needed_split}.parquet; rebuild with Parquet enabled")
    train_rows = json.loads((data_root / "train.json").read_text())
    if a.mode == "train" and len(train_rows) < a.train_batch_size:
        raise SystemExit("Training data is smaller than train-batch-size; choose a smaller compatible batch preset")
    print(json.dumps({"mode": a.mode, "variant": a.variant, "evaluation_split": split,
                      "dataset_dir": str(data_root), "reward": a.reward,
                      "data_validation": result, "output": str(out)}, indent=2))
    print(shlex.join(argv))
    if not a.run:
        return
    needed = [x for x in ("torch", "ray", "hydra", "vllm", "pyarrow") if importlib.util.find_spec(x) is None]
    if needed:
        raise SystemExit("Missing model-runtime dependencies: " + ", ".join(needed))
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < a.gpus:
        raise SystemExit(f"This preset requires {a.gpus} visible CUDA GPU(s)")
    # A fresh output directory avoids combining checkpoints or prediction manifests.
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    code_root = a.code_root.resolve() if a.code_root else ROOT / "src"
    env["PYTHONPATH"] = str(code_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONHASHSEED"] = str(a.seed)
    env.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
    env["TIMEAGENT_SEED"] = str(a.seed)
    # Keep experiment telemetry on local disk regardless of the chosen tracker:
    # wandb writes a local run dir (offline = fully local, syncable later) and
    # tensorboard event files land under the run output directory.
    env["WANDB_DIR"] = str(out)
    env.setdefault("WANDB_MODE", a.wandb_mode)
    env["TENSORBOARD_DIR"] = str(out / "tensorboard")
    manifest = {"argv": argv, "cwd": str(code_root), "seed": a.seed,
                "mode": a.mode, "variant": a.variant, "evaluation_split": split,
                "data_dir": str(data_root), "data_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    (out / "launch.json").write_text(json.dumps(manifest, indent=2))
    with (out / "console.log").open("w") as log:
        completed = subprocess.run(argv, cwd=code_root, env=env, stdout=log, stderr=subprocess.STDOUT)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
