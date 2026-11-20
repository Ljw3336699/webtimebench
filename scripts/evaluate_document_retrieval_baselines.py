#!/usr/bin/env python3
"""Document-level retrieval baselines from a frozen local index (CPU only).

Computes candidate-date quality and aggregation controls for the temporal test
split: oracle over retrieved dates, median-of-top-k, earliest/latest, and a
random-date control. No model is loaded.
"""
import argparse
import datetime
import hashlib
import json
import math
import random
import sqlite3
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from timeagent.search.local_backend import LocalCorpusBackend  # noqa: E402


def wilson_interval(hits, total, z=1.96):
    """Binomial score interval; returns None when the denominator is empty."""
    if not total:
        return None
    phat = hits / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    half = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def parse_day(value):
    try:
        return datetime.date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def utility(pred, gold):
    if pred is None:
        return 0.0
    return math.exp(-0.05 * abs((pred - gold).days) / 30.0)


def summarize(preds, golds):
    n = len(golds)
    scores = [utility(p, g) for p, g in zip(preds, golds)]
    acc7 = [0.0 if p is None else float(abs((p - g).days) <= 7) for p, g in zip(preds, golds)]
    acc30 = [0.0 if p is None else float(abs((p - g).days) <= 30) for p, g in zip(preds, golds)]
    return {"n": n, "coverage": sum(p is not None for p in preds) / n if n else 0.0,
            "score": sum(scores) / n if n else None,
            "acc_at_7": sum(acc7) / n if n else None,
            "acc_at_30": sum(acc30) / n if n else None}


def paired_bootstrap(preds, golds, metric, replicates=2000, seed=20261103):
    rng = random.Random(seed)
    n = len(golds)
    values = []
    for _ in range(replicates):
        idx = [rng.randrange(n) for _ in range(n)]
        p = [preds[i] for i in idx]
        g = [golds[i] for i in idx]
        if metric == "score":
            values.append(sum(utility(a, b) for a, b in zip(p, g)) / n)
        else:
            tol = 30 if metric == "acc_at_30" else 7
            values.append(sum(0.0 if a is None else float(abs((a - b).days) <= tol)
                              for a, b in zip(p, g)) / n)
    values.sort()
    return [values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]]


def query_of(row):
    body = row["prompt"][0]["content"].split("Content:", 1)[1]
    tokens = body.split()
    return " ".join(tokens[:12]) if tokens else body[:200]


def median_date(dates):
    if not dates:
        return None
    ordinals = sorted(d.toordinal() for d in dates)
    mid = len(ordinals) // 2
    value = ordinals[mid] if len(ordinals) % 2 else (ordinals[mid - 1] + ordinals[mid]) // 2
    return datetime.date.fromordinal(value)


def dedup_dates(results):
    dates = []
    for item in results:
        day = parse_day(item.get("publication_date"))
        if day is not None:
            dates.append(day)
    return dates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=ROOT / "webtimebench/temporal/test.json")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-date", help="Optional YYYY-MM-DD evidence cutoff")
    parser.add_argument("--random-replicates", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.split.read_text(encoding="utf-8"))
    backend = LocalCorpusBackend(args.index)
    golds = [parse_day(r["golden_answers"][0]) for r in rows]
    regimes = [r.get("difficulty", "ALL") for r in rows]

    retrieved = []
    for row in rows:
        try:
            results = backend.search(query_of(row), args.topk, max_date=args.max_date)
        except ValueError:
            results = []
        retrieved.append(dedup_dates(results))

    # Candidate-date evidence quality.
    month_total = month_hit = 0
    same_day = 0
    for dates, gold in zip(retrieved, golds):
        if not dates:
            continue
        month_total += 1
        month_hit += any((d.year, d.month) == (gold.year, gold.month) for d in dates)
        same_day += any(d == gold for d in dates)

    # The hit criterion above is (year, month), so a uniform 1/12 calendar-month
    # baseline is the wrong comparison. The matched baseline is the collision rate
    # of the split's own gold year-month distribution: sum_m p_m^2.
    gold_month_counts = Counter((g.year, g.month) for g in golds)
    year_month_collision = sum((c / len(golds)) ** 2 for c in gold_month_counts.values())
    calendar_month_counts = Counter(g.month for g in golds)
    calendar_month_collision = sum((c / len(golds)) ** 2 for c in calendar_month_counts.values())

    dates_pool = [d for dates in retrieved for d in dates]
    rng = random.Random(20261103)

    baselines = {}
    for k in range(1, args.topk + 1):
        preds = [median_date(dates[:k]) for dates in retrieved]
        baselines[f"median_top{k}"] = summarize(preds, golds)
        baselines[f"median_top{k}"]["ci_score"] = paired_bootstrap(preds, golds, "score")
        baselines[f"median_top{k}"]["ci_acc30"] = paired_bootstrap(preds, golds, "acc_at_30")

    baselines["earliest"] = summarize([min(d) if d else None for d in retrieved], golds)
    baselines["latest"] = summarize([max(d) if d else None for d in retrieved], golds)

    oracle = []
    for dates, gold in zip(retrieved, golds):
        oracle.append(min(dates, key=lambda d: abs((d - gold).days)) if dates else None)
    baselines["oracle_best_retrieved"] = summarize(oracle, golds)
    baselines["oracle_best_retrieved"]["ci_acc30"] = paired_bootstrap(oracle, golds, "acc_at_30")

    random_preds = []
    if dates_pool:
        for _ in range(len(rows)):
            picks = [rng.choice(dates_pool) for _ in range(args.random_replicates)]
            random_preds.append(median_date(picks))
    else:
        random_preds = [None] * len(rows)
    baselines["random_retrieved_date"] = summarize(random_preds, golds)
    baselines["random_retrieved_date"]["ci_score"] = paired_bootstrap(random_preds, golds, "score")

    per_regime = {}
    for regime in sorted(set(regimes)):
        idx = [i for i, r in enumerate(regimes) if r == regime]
        per_regime[regime] = {
            "n": len(idx),
            "oracle_acc_at_30": sum(0.0 if oracle[i] is None else float(abs((oracle[i] - golds[i]).days) <= 30)
                                    for i in idx) / len(idx),
            "median_top3": summarize([median_date(retrieved[i][:3]) for i in idx],
                                     [golds[i] for i in idx]),
            "queries_with_evidence": sum(bool(retrieved[i]) for i in idx) / len(idx),
        }

    report = {
        "schema": "timebench.document_retrieval_baselines.v1",
        "index": str(args.index.resolve()),
        "index_sha256": hashlib.sha256(args.index.read_bytes()).hexdigest(),
        "split": str(args.split),
        "split_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest(),
        "n": len(rows),
        "topk": args.topk,
        "max_date": args.max_date,
        "query_definition": "first 12 whitespace tokens after the first 'Content:'",
        "evidence_quality": {
            "queries_with_any_dated_hit": month_total,
            "gold_month_shared_by_any_hit": month_hit,
            "gold_month_match_definition": "(year, month) equality with any retrieved date",
            "gold_month_rate": month_hit / month_total if month_total else None,
            "gold_month_rate_denominator": month_total,
            "gold_month_rate_full_denominator": month_hit / len(rows) if rows else None,
            "gold_month_rate_ci95": wilson_interval(month_hit, month_total),
            "gold_month_rate_full_ci95": wilson_interval(month_hit, len(rows)),
            "random_month_rate": 1 / 12,
            "random_month_rate_definition": "uniform calendar month; retained for continuity "
                                            "and NOT the matched baseline for a (year, month) criterion",
            "random_month_rate_empirical_year_month": year_month_collision,
            "random_month_rate_empirical_calendar_month": calendar_month_collision,
            "gold_day_shared_by_any_hit": same_day,
        },
        "baselines": baselines,
        "per_regime": per_regime,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["evidence_quality"], indent=2))
    for name in ("oracle_best_retrieved", "median_top3", "random_retrieved_date"):
        print(name, json.dumps(baselines[name], indent=2))


if __name__ == "__main__":
    main()
