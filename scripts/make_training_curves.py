#!/usr/bin/env python3
"""Plot dev Score across training runs from per-step validation metrics.

Each `--run label=path` points at a launcher output directory; the script reads
`predictions/step_*/evaluation_*.metrics.json` written by dev validation.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_run(run_dir, metric):
    points = {}
    for path in sorted((Path(run_dir) / "predictions").glob("step_*/evaluation_*.metrics.json")):
        suffix = path.parent.name.split("_", 1)[-1]
        if not suffix.isdigit():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8")).get("metrics", {})
        if metric not in metrics:
            continue
        points[int(suffix)] = {
            "score": metrics[metric],
            "score_smooth_macro": metrics.get("score_macro"),
            "acc_at_30": metrics.get("acc_at_30"),
            "search_turns": metrics.get("avg_search_turns"),
        }
    return sorted(points.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="label=run_directory")
    parser.add_argument("--metric", default="score_micro")
    parser.add_argument("--degrade-from", type=int, default=200,
                        help="Start of the shaded late-training degradation period")
    parser.add_argument("--output", type=Path,
                        default=Path("paper/figures/fig1_training_curves.pdf"))
    args = parser.parse_args()

    figure, axis = plt.subplots(figsize=(6, 3.2))
    summary = {}
    last_step = 0
    for spec in args.run:
        label, run_dir = spec.split("=", 1)
        points = load_run(run_dir, args.metric)
        if not points:
            print(f"skip {label}: no {args.metric} metrics under {run_dir}")
            continue
        steps = [step for step, _ in points]
        scores = [value["score"] for _, value in points]
        axis.plot(steps, scores, marker="o", markersize=3, linewidth=1.5, label=label)
        best = max(points, key=lambda item: item[1]["score"])
        summary[label] = {
            "steps": steps,
            "score": scores,
            "best_step": best[0],
            "best_score": best[1]["score"],
            "best_acc_at_30": best[1]["acc_at_30"],
            "final_score": scores[-1],
        }
        last_step = max(last_step, steps[-1])

    if last_step > args.degrade_from:
        axis.axvspan(args.degrade_from, last_step, color="red", alpha=0.08,
                     label="late-training degradation")
    axis.set_xlabel("PPO step")
    axis.set_ylabel("Dev Score")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)
    figure.savefig(args.output.with_suffix(".png"), dpi=200)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
