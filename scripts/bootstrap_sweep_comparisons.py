#!/usr/bin/env python3
"""Paired bootstrap on Acc@30 for the evaluation-only sweep.

Compares, per untrained model, real evidence against no evidence and against random retrieval, and each
trained policy against the untrained Qwen2.5-7B-Instruct under real evidence. Reads the per-example
prediction files the evaluators write (`predictions/step_0/evaluation_*.jsonl`); an invalid or missing
prediction counts as a miss. Resamples examples with replacement (2,000 draws, seed 20261103).
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import random
from pathlib import Path

MODELS = {"q7i": "Qwen2.5-7B-Instruct", "q3i": "Qwen2.5-3B-Instruct", "l3i": "Llama-3.2-3B-Instruct",
          "r1d": "DeepSeek-R1-Distill-Qwen-7B", "tr1": "Time-R1-Theta2", "qw3": "Qwen3-8B (standalone client)"}
POLICIES = {"swp_pol_ref100_real": "ref100", "swp_pol_ref150_real": "ref150",
            "swp_pol_A150_real": "A150 (cost 0)", "swp_pol_C150_real": "C150 (entropy 0.01)"}


def load(runs, name):
    files = sorted(glob.glob(str(runs / name / "predictions" / "step_0" / "evaluation_*.jsonl")))
    if not files:
        raise SystemExit(f"no predictions for {name}")
    return {row["id"]: row for row in map(json.loads, Path(files[-1]).read_text().splitlines())}


def hit(row, days=30):
    try:
        p = dt.date.fromisoformat(str(row["prediction"])[:10])
        g = dt.date.fromisoformat(str(row["gold"])[:10])
    except (TypeError, ValueError):
        return 0
    return int(abs((p - g).days) <= days)


def paired(a, b, draws=2000, seed=20261103):
    ids = sorted(set(a) & set(b))
    diff = [hit(a[i]) - hit(b[i]) for i in ids]
    rng, n = random.Random(seed), len(diff)
    means = sorted(sum(diff[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return {"n": n, "mean_diff": sum(diff) / n, "ci95": [means[int(.025 * draws)], means[int(.975 * draws) - 1]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = {"metric": "Acc@30 (invalid counts as miss), paired bootstrap, 2000 draws, seed 20261103",
           "real_minus_null": {}, "real_minus_random": {}, "policy_minus_untrained_q7i_real": {}}
    for tag, name in MODELS.items():
        real, null, rand = (load(args.runs, f"swp_{tag}_{c}") for c in ("real", "null", "random"))
        out["real_minus_null"][name] = paired(real, null)
        out["real_minus_random"][name] = paired(real, rand)
    base = load(args.runs, "swp_q7i_real")
    for run, name in POLICIES.items():
        out["policy_minus_untrained_q7i_real"][name] = paired(load(args.runs, run), base)
    text = json.dumps(out, indent=1)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
