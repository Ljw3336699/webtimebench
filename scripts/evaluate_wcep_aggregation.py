#!/usr/bin/env python3
"""End-to-end WCEP event-level retrieval aggregation evaluation.

Compares:
  (a) oracle same-event aggregation (upper bound, uses gold event ids)
  (b) BM25 retrieval aggregation (frozen local index queried by article title)
  (c) same-event recall of BM25 top-k
All date errors are in days; Acc@k uses the same full-denominator convention.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from timeagent.search.local_backend import LocalCorpusBackend  # noqa: E402


def parse_date(value):
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def load_articles(path):
    rows = []
    for line in Path(path).open(encoding="utf-8"):
        row = json.loads(line)
        day = parse_date(row.get("date") or row.get("article_time_raw")
                         or row.get("publication_date_candidate"))
        if day is None:
            continue
        rows.append({"id": row["id"], "event": row["event_id"],
                     "title": row.get("title") or "", "day": day.toordinal(),
                     "date": day.isoformat()})
    return rows


def acc(actual, pred, tol):
    return 0.0 if pred is None else float(abs(actual - pred) <= tol)


def evaluate(articles, backend, topk=3):
    by_event = defaultdict(list)
    for i, article in enumerate(articles):
        by_event[article["event"]].append(i)

    oracle = {0: [], 7: [], 30: []}
    for i, article in enumerate(articles):
        same = [articles[j] for j in by_event[article["event"]] if j != i]
        if not same:
            continue
        pred = statistics.median([x["day"] for x in same])
        for tol in (0, 7, 30):
            oracle[tol].append(acc(article["day"], pred, tol))

    recall_hit = recall_total = 0
    agg = {0: [], 7: [], 30: []}
    per_case = []
    for i, article in enumerate(articles):
        try:
            results = backend.search(article["title"], topk)
        except ValueError:
            continue
        others = [r for r in results if r.get("corpus_id") != article["id"]
                  and parse_date(r.get("publication_date")) is not None][:topk]
        recall_total += 1
        ids = {articles[j]["id"] for j in by_event[article["event"]] if j != i}
        if any(r.get("corpus_id") in ids for r in others):
            recall_hit += 1
        # Full denominator: no retrievable other article counts as a failure, never skipped.
        pred = statistics.median([parse_date(r.get("publication_date")).toordinal() for r in others]) if others else None
        for tol in (0, 7, 30):
            agg[tol].append(acc(article["day"], pred, tol))
        per_case.append({"id": article["id"], "event": article["event"],
                         "gold": article["date"], "predicted_ordinal": int(pred),
                         "same_event_in_topk": any(r.get("corpus_id") in ids for r in others),
                         "retrieved": [{"corpus_id": r.get("corpus_id"),
                                        "date": r.get("publication_date")} for r in others]})
    return {
        "oracle": {f"acc_at_{k}": round(statistics.mean(v), 4) if v else None
                   for k, v in oracle.items()},
        "bm25_topk": topk,
        "bm25_recall_same_event": round(recall_hit / max(recall_total, 1), 4),
        "bm25_recall_denominator": recall_total,
        "bm25_aggregate": {f"acc_at_{k}": round(statistics.mean(v), 4) if v else None
                           for k, v in agg.items()},
        "cases": per_case,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", type=Path, required=True,
                        help="WCEP pilot articles.jsonl (gold ids/dates/events)")
    parser.add_argument("--index", type=Path, required=True,
                        help="Frozen local BM25 index built from the converted corpus")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    articles = load_articles(args.articles)
    backend = LocalCorpusBackend(args.index)
    result = evaluate(articles, backend, topk=args.topk)
    result.update({"articles": len(articles), "index": args.index.resolve().as_posix(),
                   "backend_id": backend.backend_id, "topk": args.topk})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()