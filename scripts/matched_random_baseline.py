#!/usr/bin/env python3
"""Random-retrieval baseline matched to the any-of-k year-month hit criterion (CPU only).

`evaluate_document_retrieval_baselines.py` counts a query as a hit when ANY of its retrieved dated
documents shares the gold (year, month). A single-draw collision rate (sum_m p_m^2) is the right
baseline only for one retrieved date. For a query that retrieved m dated documents, random retrieval
from the same index hits with probability 1 - C(N-K, m) / C(N, m), where N is the number of dated
indexed documents (inside the window filter, if any) and K those in the gold month. This script
re-runs the same retrieval, then reports the observed rate next to that expectation, averaged over the
same queries, so the comparison uses the same queries, the same retrieval budget and the same index.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_document_retrieval_baselines as base  # noqa: E402


def hit_probability(n_total, n_month, m):
    if m <= 0 or n_total <= 0:
        return 0.0
    if m > n_total:
        m = n_total
    if n_total - n_month < m:
        return 1.0
    miss = 1.0
    for j in range(m):
        miss *= (n_total - n_month - j) / (n_total - j)
    return 1.0 - miss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=ROOT / "webtimebench/temporal/test.json")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-date")
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.split.read_text(encoding="utf-8"))
    golds = [base.parse_day(r["golden_answers"][0]) for r in rows]
    backend = base.LocalCorpusBackend(args.index)

    connection = sqlite3.connect(f"file:{args.index}?mode=ro", uri=True)
    cutoff = base.parse_day(args.max_date) if args.max_date else None
    pool = []
    for (record,) in connection.execute("SELECT record FROM documents"):
        day = base.parse_day(json.loads(record).get("publication_date"))
        if day is not None and (cutoff is None or day <= cutoff):
            pool.append((day.year, day.month))
    month_counts = Counter(pool)
    n_total = len(pool)

    rng = random.Random(args.seed)
    observed = expected = simulated = 0.0
    with_evidence = 0
    sizes = Counter()
    for row, gold in zip(rows, golds):
        try:
            results = backend.search(base.query_of(row), args.topk, max_date=args.max_date)
        except ValueError:
            results = []
        dates = base.dedup_dates(results)
        if not dates:
            continue
        with_evidence += 1
        sizes[len(dates)] += 1
        key = (gold.year, gold.month)
        observed += any((d.year, d.month) == key for d in dates)
        expected += hit_probability(n_total, month_counts.get(key, 0), len(dates))
        simulated += sum(any(m == key for m in rng.sample(pool, min(len(dates), n_total)))
                         for _ in range(args.replicates)) / args.replicates

    result = {
        "index_sha256": __import__("hashlib").sha256(args.index.read_bytes()).hexdigest(),
        "split": str(args.split.relative_to(ROOT)) if args.split.is_relative_to(ROOT) else args.split.name,
        "n_queries": len(rows), "queries_with_dated_hit": with_evidence, "topk": args.topk,
        "max_date": args.max_date, "dated_indexed_documents": n_total,
        "dated_documents_in_gold_months_share": sum(month_counts.get((g.year, g.month), 0) for g in golds)
        / (len(golds) * n_total) if n_total else None,
        "retrieved_dates_per_query": dict(sorted(sizes.items())),
        "observed_rate": observed / with_evidence if with_evidence else None,
        "random_expected_rate_same_budget": expected / with_evidence if with_evidence else None,
        "random_simulated_rate_same_budget": simulated / with_evidence if with_evidence else None,
        "single_draw_collision_rate_of_gold": sum((c / len(golds)) ** 2 for c in
                                                  Counter((g.year, g.month) for g in golds).values()),
        "observed_minus_random_expected": (observed - expected) / with_evidence if with_evidence else None,
        "definition": "hit = any retrieved dated document shares the gold (year, month); random = the same "
                      "number of dated documents drawn from the same index without replacement",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
