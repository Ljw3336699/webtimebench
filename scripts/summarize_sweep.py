#!/usr/bin/env python3
"""Collect the evaluation-only sweep into one table: model x evidence condition."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

NAMES = {"q7i": "Qwen2.5-7B-Instruct", "q3i": "Qwen2.5-3B-Instruct", "q3b": "Qwen2.5-3B (base)",
         "l3i": "Llama-3.2-3B-Instruct", "r1d": "DeepSeek-R1-Distill-Qwen-7B", "tr1": "Time-R1-Theta2 (3B)",
         "q7s": "Qwen2.5-7B-Instruct (standalone client)", "qw3": "Qwen3-8B (standalone, no-think)"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    for d in sorted(glob.glob(str(args.runs / "swp_*"))):
        files = glob.glob(d + "/predictions/step_0/evaluation_*.metrics.json")
        if not files:
            continue
        m = json.loads(Path(files[0]).read_text())["metrics"]
        parts = Path(d).name.split("_")
        who = NAMES.get(parts[1], "policy " + "_".join(parts[2:-1])) if parts[1] != "pol" else "policy " + parts[2]
        cond = parts[-1]
        rows.append({"model": who, "condition": cond, "n": int(m["count"]), "coverage": round(m["prediction_coverage"], 3),
                     "score": round(m["score_micro"], 4), "acc_at_30": round(m["acc_at_30"], 4),
                     "search_rate": round(m["search_rate"], 3)})
    print(f"{'model':32s} {'evidence':8s} {'cov':>6s} {'Score':>7s} {'Acc@30':>7s} {'search':>7s}")
    for r in rows:
        print(f"{r['model']:32s} {r['condition']:8s} {r['coverage']:6.3f} {r['score']:7.4f} {r['acc_at_30']:7.4f} {r['search_rate']:7.3f}")
    if args.output:
        args.output.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
