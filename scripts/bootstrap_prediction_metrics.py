#!/usr/bin/env python3
"""Bootstrap metrics for a prediction JSONL (full denominator, missing=failure)."""
import argparse
import datetime
import json
import math
import random
from pathlib import Path


def parse_day(value):
    try:
        return datetime.date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def utility(pred, gold):
    return 0.0 if pred is None else math.exp(-0.05 * abs((pred - gold).days) / 30.0)


def metrics(rows):
    n = len(rows)
    scores = [utility(parse_day(r.get("prediction")), parse_day(r.get("gold"))) for r in rows]
    acc7 = []
    acc30 = []
    for r in rows:
        p, g = parse_day(r.get("prediction")), parse_day(r.get("gold"))
        acc7.append(0.0 if p is None else float(abs((p - g).days) <= 7))
        acc30.append(0.0 if p is None else float(abs((p - g).days) <= 30))
    return {"n": n, "coverage": sum(parse_day(r.get("prediction")) is not None for r in rows) / n,
            "score": sum(scores) / n, "acc_at_7": sum(acc7) / n, "acc_at_30": sum(acc30) / n}


def bootstrap(rows, metric, replicates, seed):
    rng = random.Random(seed)
    n = len(rows)
    stats = []
    for _ in range(replicates):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        stats.append(metrics(sample)[metric])
    stats.sort()
    return [stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats)) - 1]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = {"schema": "timebench.bootstrap_metrics.v1", "predictions": str(args.predictions),
              "n": len(rows), "overall": metrics(rows), "per_regime": {}}
    for metric in ("score", "acc_at_30"):
        report["overall"][f"ci_{metric}"] = bootstrap(rows, metric, args.replicates, args.seed)
    for regime in sorted({r.get("regime", "ALL") for r in rows}):
        subset = [r for r in rows if r.get("regime", "ALL") == regime]
        entry = metrics(subset)
        for metric in ("score", "acc_at_30"):
            entry[f"ci_{metric}"] = bootstrap(subset, metric, args.replicates, args.seed)
        report["per_regime"][regime] = entry
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
