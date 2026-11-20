#!/usr/bin/env python3
"""Figure 1: dev Score and executed-search rate against training step for one run.

Reads only the per-step validation manifests the launcher writes
(`predictions/step_*/evaluation_*.metrics.json`), so the figure is regenerated from
the same artifacts the metrics ledger cites. Marks the dev-selected checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(run_dir):
    points = []
    for path in (Path(run_dir) / "predictions").glob("step_*/evaluation_*.metrics.json"):
        step = int(path.parent.name.split("_", 1)[1])
        metrics = json.loads(path.read_text())["metrics"]
        points.append((step, metrics["score_micro"], metrics["score_macro"],
                       metrics["search_rate"], metrics["avg_response_tokens"]))
    return sorted(points)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    points = load(args.run)
    if not points:
        raise SystemExit(f"no validation manifests under {args.run}")
    steps = [p[0] for p in points]
    micro = [p[1] for p in points]
    search = [p[3] for p in points]
    best = max(points, key=lambda p: p[1])[0]

    fig, left = plt.subplots(figsize=(3.4, 2.1))
    left.plot(steps, micro, marker="o", ms=3, lw=1.4, color="#1f4e79", label="dev Score")
    left.set_xlabel("training step")
    left.set_ylabel("dev Score", color="#1f4e79")
    left.tick_params(axis="y", labelcolor="#1f4e79")
    right = left.twinx()
    right.plot(steps, search, marker="s", ms=3, lw=1.4, ls="--", color="#b5532b",
               label="dev search rate")
    right.set_ylabel("search rate", color="#b5532b")
    right.tick_params(axis="y", labelcolor="#b5532b")
    right.set_ylim(-0.03, 0.62)
    left.axvline(best, color="0.45", lw=0.8, ls=":")
    left.annotate(f"dev peak\n(step {best})", xy=(best, max(micro)), xytext=(4, -18),
                  textcoords="offset points", fontsize=6.5, color="0.3")
    for axis in (left, right):
        axis.tick_params(labelsize=7)
        axis.xaxis.label.set_size(7.5)
        axis.yaxis.label.set_size(7.5)
    fig.tight_layout(pad=0.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    fig.savefig(args.output.with_suffix(".png"), dpi=200)
    print(json.dumps({"run": str(args.run), "steps": steps, "score_micro": micro,
                      "search_rate": search, "dev_selected_step": best}))


if __name__ == "__main__":
    main()
