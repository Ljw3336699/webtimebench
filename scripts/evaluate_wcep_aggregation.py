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
import hashlib
import json
import os
import random
import re
import statistics
import sys
import unicodedata
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from timeagent.search.local_backend import LocalCorpusBackend  # noqa: E402


def normalize_title(value):
    """Same normalization the event clustering uses: NFKC, casefold, alphanumeric only."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.sub(r"[^0-9a-z\u3400-\u9fff]+", " ", text).split())


def parse_date(value):
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def load_articles(path, query_field="title"):
    rows = []
    for line in Path(path).open(encoding="utf-8"):
        row = json.loads(line)
        day = parse_date(row.get("date") or row.get("article_time_raw")
                         or row.get("publication_date_candidate"))
        if day is None:
            continue
        query = row.get("title") or ""
        if query_field == "body":
            query = " ".join((row.get("text") or row.get("content") or "").split()[:12])
        body = " ".join((row.get("text") or row.get("content") or "").split())
        rows.append({"id": row["id"], "event": row["event_id"],
                     "title": query, "title_norm": normalize_title(row.get("title")),
                     "body_hash": hashlib.sha256(body.encode()).hexdigest(),
                     "day": day.toordinal(), "date": day.isoformat()})
    return rows


def acc(actual, pred, tol):
    return 0.0 if pred is None else float(abs(actual - pred) <= tol)


def evaluate(articles, backend, topk=3, exclude_identical_body=False,
             exclude_exact_title=False, retrieval="bm25", seed=20261119,
             shard=0, num_shards=1):
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

    rng = random.Random(seed)
    recall_hit = recall_total = recall_hit_filtered = dropped_exact_title = 0
    agg = {0: [], 7: [], 30: []}
    # Same retrieval, evaluated with and without exact-title near-duplicates, so the
    # leakage control costs no extra pass over the corpus.
    agg_notitle = {0: [], 7: [], 30: []}
    recall_hit_notitle = 0
    per_case = []
    shard_indexes = range(shard, len(articles), num_shards)
    for i in shard_indexes:
        article = articles[i]
        if recall_total and not recall_total % 2000:
            print(f"[progress] shard {shard}/{num_shards}: {recall_total} evaluated",
                  file=sys.stderr, flush=True)
        if retrieval == "random":
            # Control: same denominator and top-k budget, no lexical signal.
            picks = [articles[rng.randrange(len(articles))] for _ in range(topk)]
            results = [{"corpus_id": p["id"], "title": p["title_norm"],
                        "publication_date": p["date"]} for p in picks]
        else:
            try:
                results = backend.search(article["title"], topk)
            except ValueError:
                # Invalid/empty queries stay in the denominator as failures.
                results = []
        others = [r for r in results if r.get("corpus_id") != article["id"]
                  and parse_date(r.get("publication_date")) is not None]
        notitle = [r for r in others
                   if normalize_title(r.get("title")) != article["title_norm"]]
        dropped_exact_title += len(others) - len(notitle)
        if exclude_exact_title:
            others = notitle
        if exclude_identical_body:
            others = [r for r in others
                      if hashlib.sha256(" ".join(str(r.get("content") or "").split()).encode()).hexdigest()
                      != article["body_hash"]]
        others = others[:topk]
        recall_total += 1
        ids = {articles[j]["id"] for j in by_event[article["event"]] if j != i}
        if any(r.get("corpus_id") in ids for r in results):
            recall_hit += 1
        if any(r.get("corpus_id") in ids for r in others):
            recall_hit_filtered += 1
        # Full denominator: no retrievable other article counts as a failure, never skipped.
        pred = statistics.median([parse_date(r.get("publication_date")).toordinal() for r in others]) if others else None
        for tol in (0, 7, 30):
            agg[tol].append(acc(article["day"], pred, tol))
        notitle = notitle[:topk]
        if any(r.get("corpus_id") in ids for r in notitle):
            recall_hit_notitle += 1
        pred_notitle = statistics.median(
            [parse_date(r.get("publication_date")).toordinal() for r in notitle]) if notitle else None
        for tol in (0, 7, 30):
            agg_notitle[tol].append(acc(article["day"], pred_notitle, tol))
        per_case.append({"id": article["id"], "event": article["event"],
                         "gold": article["date"],
                         "predicted_ordinal": pred if pred is None else round(float(pred), 3),
                         "same_event_in_topk": any(r.get("corpus_id") in ids for r in others),
                         "retrieved": [{"corpus_id": r.get("corpus_id"),
                                        "date": r.get("publication_date")} for r in others]})
    return {
        "oracle": {f"acc_at_{k}": round(statistics.mean(v), 4) if v else None
                   for k, v in oracle.items()},
        "oracle_denominator": len(oracle[0]),
        "bm25_topk": topk,
        "bm25_recall_same_event": round(recall_hit / max(recall_total, 1), 4),
        "bm25_recall_same_event_after_filters": round(recall_hit_filtered / max(recall_total, 1), 4),
        "bm25_recall_denominator": recall_total,
        "recall_definitions": {
            "bm25_recall_same_event": "any same-event id in the raw top-k list",
            "bm25_recall_same_event_after_filters":
                "any same-event id after self-exclusion, date-parse and the enabled "
                "exact-title/identical-body filters; this matches the pre-2026-11-19 "
                "definition of bm25_recall_same_event",
        },
        "exact_title_hits_dropped": dropped_exact_title,
        "bm25_aggregate": {f"acc_at_{k}": round(statistics.mean(v), 4) if v else None
                           for k, v in agg.items()},
        "exact_title_excluded": {
            "recall_same_event": round(recall_hit_notitle / max(recall_total, 1), 4),
            "aggregate": {f"acc_at_{k}": round(statistics.mean(v), 4) if v else None
                          for k, v in agg_notitle.items()},
            "note": "same retrieval as above, with retrieved documents whose normalized "
                    "title equals the query article's title removed",
        },
        "shard": {"index": shard, "num_shards": num_shards},
        "sums": {
            "n": recall_total,
            "recall_hit_raw": recall_hit,
            "recall_hit_after_filters": recall_hit_filtered,
            "recall_hit_exact_title_excluded": recall_hit_notitle,
            "dropped_exact_title": dropped_exact_title,
            "hits": {f"acc_at_{k}": int(sum(v)) for k, v in agg.items()},
            "hits_exact_title_excluded": {f"acc_at_{k}": int(sum(v)) for k, v in agg_notitle.items()},
        },
        "cases": per_case,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", type=Path, required=True,
                        help="WCEP pilot articles.jsonl (gold ids/dates/events)")
    parser.add_argument("--index", type=Path, required=True,
                        help="Frozen local BM25 index built from the converted corpus")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--query-field", choices=("title", "body"), default="title")
    parser.add_argument("--exclude-identical-body", action="store_true")
    parser.add_argument("--exclude-exact-title", action="store_true",
                        help="Drop retrieved documents whose normalized title equals the query "
                             "article's title (removes title near-duplicate retrieval)")
    parser.add_argument("--retrieval", choices=("bm25", "random"), default="bm25",
                        help="bm25 = frozen index; random = topk random dated articles (control)")
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Evaluate articles[shard::num_shards]; merge the shard JSONs by summing sums{}")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    articles = load_articles(args.articles, args.query_field)
    backend = LocalCorpusBackend(args.index)
    result = evaluate(articles, backend, topk=args.topk,
                      exclude_identical_body=args.exclude_identical_body,
                      exclude_exact_title=args.exclude_exact_title,
                      retrieval=args.retrieval, seed=args.seed,
                      shard=args.shard, num_shards=args.num_shards)
    result.update({"articles": len(articles), "index": args.index.resolve().as_posix(),
                   "backend_id": backend.backend_id, "topk": args.topk,
                   "config": {"query_field": args.query_field, "retrieval": args.retrieval,
                              "exclude_exact_title": args.exclude_exact_title,
                              "exclude_identical_body": args.exclude_identical_body,
                              "seed": args.seed}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()