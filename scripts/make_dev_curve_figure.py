#!/usr/bin/env python3
"""Figure 1: dev Score and dev search rate against training step, one line per run.

Reads only the per-step validation manifests the launcher writes
(`predictions/step_*/evaluation_*.metrics.json`), so the figure regenerates from the same
artifacts the paper cites. Usage:
  make_dev_curve_figure.py --run "cost 0.01=outputs/seed_20261103" --run "cost 0.1=outputs/..." --output fig.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLES = [("#b5532b", "-", "o"), ("#1f4e79", "-", "s"), ("#4a8f4a", "--", "^"),
          ("#7a4fa3", "--", "D"), ("#888888", ":", "v")]


def load(run_dir, max_step):
    points = []
    for path in (Path(run_dir) / "predictions").glob("step_*/evaluation_*.metrics.json"):
        step = int(path.parent.name.split("_", 1)[1])
        if max_step is not None and step > max_step:
            continue
        metrics = json.loads(path.read_text())["metrics"]
        points.append((step, metrics["score_micro"], metrics["search_rate"]))
    return sorted(points)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="label=run_directory")
    parser.add_argument("--max-step", type=int, default=None, help="drop validation points beyond this step")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(3.4, 3.0), sharex=True)
    summary = {}
    for i, spec in enumerate(args.run):
        label, run_dir = spec.split("=", 1)
        points = load(run_dir, args.max_step)
        if not points:
            raise SystemExit(f"no validation manifests under {run_dir}")
        color, line, marker = STYLES[i % len(STYLES)]
        steps = [p[0] for p in points]
        top.plot(steps, [p[1] for p in points], color=color, ls=line, marker=marker, ms=2.5, lw=1.2, label=label)
        bottom.plot(steps, [p[2] for p in points], color=color, ls=line, marker=marker, ms=2.5, lw=1.2)
        summary[label] = {"steps": steps, "score_micro": [round(p[1], 4) for p in points],
                          "search_rate": [round(p[2], 4) for p in points]}
    top.set_ylabel("dev Score", fontsize=7.5)
    bottom.set_ylabel("dev search rate", fontsize=7.5)
    bottom.set_xlabel("training step", fontsize=7.5)
    bottom.set_ylim(-0.03, 1.05)
    for axis in (top, bottom):
        axis.tick_params(labelsize=7)
        axis.grid(alpha=0.25, lw=0.5)
    top.legend(fontsize=6, loc="lower right", frameon=False)
    fig.tight_layout(pad=0.3, h_pad=0.4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    fig.savefig(args.output.with_suffix(".png"), dpi=200)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
