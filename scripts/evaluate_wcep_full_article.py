#!/usr/bin/env python3
"""Article-level event aggregation on the full WCEP corpus (same protocol as the
WCEP pilot and WebNewsEvents): each article's gold is its own parsed publication
date; aggregation uses the median of eligible non-self dated hits among the raw
top-k (same-event recall is measured separately and excludes the query article).
Only articles with a parseable publication date are evaluated (denominator
reported separately).
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
    """Same normalization the event clustering uses: NFKC, casefold, alphanumeric only.

    Titles that collapse to the same string are near-duplicates (e.g. a curly vs
    straight apostrophe), which is exactly the signal the clustering is built on.
    """
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.sub(r"[^0-9a-z\u3400-\u9fff]+", " ", text).split())


def parse_date(value):
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def acc(actual, pred, tol):
    return 0.0 if pred is None else float(abs(actual - pred) <= tol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--query-field", choices=("title", "body"), default="title",
                        help="Use the article title or the first 12 body tokens as query")
    parser.add_argument("--exclude-identical-body", action="store_true",
                        help="Drop retrieved documents whose normalized body equals the query body")
    parser.add_argument("--exclude-exact-title", action="store_true",
                        help="Drop retrieved documents whose normalized title equals the query "
                             "article's title (removes title near-duplicate retrieval)")
    parser.add_argument("--retrieval", choices=("bm25", "random"), default="bm25",
                        help="bm25 = frozen index; random = topk random dated articles (control)")
    parser.add_argument("--seed", type=int, default=20261119, help="Seed for --retrieval random")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Evaluate rows[shard::num_shards]; merge shard JSONs by summing sums{}")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")

    rows = []
    by_id = {}
    for line in Path(args.corpus).open(encoding="utf-8"):
        r = json.loads(line)
        d = parse_date(r.get("publication_date"))
        query = r.get("title") or ""
        if args.query_field == "body":
            query = " ".join((r.get("content") or "").split()[:12])
        body_hash = hashlib.sha256(" ".join((r.get("content") or "").split()).encode()).hexdigest()
        rows.append({"id": r["corpus_id"], "event": r["event_id"], "title": query,
                     "title_norm": normalize_title(r.get("title")),
                     "body_hash": body_hash, "day": d.toordinal() if d else None})
        by_id[r["corpus_id"]] = r
    by_event = defaultdict(list)
    for r in rows:
        by_event[r["event"]].append(r)

    backend = LocalCorpusBackend(args.index)
    rng = random.Random(args.seed)
    dated_rows = [r for r in rows if r["day"] is not None]
    stats = {0: [], 7: [], 30: []}
    stats_notitle = {0: [], 7: [], 30: []}
    recall_hit_notitle = 0
    total = recall_hit = recall_den = 0
    dropped_exact_title = 0
    recall_hit_filtered = 0
    for position in range(args.shard, len(rows), args.num_shards):
        r = rows[position]
        if total and not total % 2000:
            print(f"[progress] shard {args.shard}/{args.num_shards}: {total} evaluated",
                  file=sys.stderr, flush=True)
        if r["day"] is None:
            continue  # gold unavailable; other articles' dates still usable as evidence
        if args.retrieval == "random":
            # Control: same denominator and same top-k budget, no lexical signal.
            picks = [dated_rows[rng.randrange(len(dated_rows))] for _ in range(args.topk)]
            results = [{"corpus_id": p["id"], "title": p["title_norm"],
                        "publication_date": date.fromordinal(p["day"]).isoformat()}
                       for p in picks]
        else:
            try:
                results = backend.search(str(r["title"]), args.topk)
            except ValueError:
                results = []
        others = [x for x in results if x.get("corpus_id") != r["id"]
                  and parse_date(x.get("publication_date")) is not None]
        notitle = [x for x in others if normalize_title(x.get("title")) != r["title_norm"]]
        dropped_exact_title += len(others) - len(notitle)
        if args.exclude_exact_title:
            others = notitle
        if args.exclude_identical_body:
            others = [x for x in others
                      if hashlib.sha256(" ".join(str(x.get("contents") or x.get("content") or "").split())
                                        .encode()).hexdigest() != r["body_hash"]]
        others = others[:args.topk]
        total += 1
        own = set(x["id"] for x in by_event[r["event"]] if x["id"] != r["id"])
        recall_den += 1
        if any(x.get("corpus_id") in own for x in results):
            recall_hit += 1
        if any(x.get("corpus_id") in own for x in others):
            recall_hit_filtered += 1
        pred = statistics.median([parse_date(x.get("publication_date")).toordinal() for x in others]) if others else None
        for tol in (0, 7, 30):
            stats[tol].append(acc(r["day"], pred, tol))
        notitle = notitle[:args.topk]
        if any(x.get("corpus_id") in own for x in notitle):
            recall_hit_notitle += 1
        pred_nt = statistics.median(
            [parse_date(x.get("publication_date")).toordinal() for x in notitle]) if notitle else None
        for tol in (0, 7, 30):
            stats_notitle[tol].append(acc(r["day"], pred_nt, tol))
    result = {"articles_evaluated": total, "topk": args.topk,
              "backend": backend.backend_id,
              "config": {"query_field": args.query_field, "retrieval": args.retrieval,
                         "exclude_exact_title": args.exclude_exact_title,
                         "exclude_identical_body": args.exclude_identical_body,
                         "seed": args.seed},
              "exact_title_hits_dropped": dropped_exact_title,
              "same_event_recall_topk": round(recall_hit / max(recall_den, 1), 4),
              "same_event_recall_after_filters": round(recall_hit_filtered / max(recall_den, 1), 4),
              "aggregate": {f"acc_at_{k}": round(statistics.mean(v), 4) for k, v in stats.items()},
              "exact_title_excluded": {
                  "recall_same_event": round(recall_hit_notitle / max(recall_den, 1), 4),
                  "aggregate": {f"acc_at_{k}": round(statistics.mean(v), 4)
                                for k, v in stats_notitle.items()}},
              "shard": {"index": args.shard, "num_shards": args.num_shards},
              "sums": {"n": total, "recall_hit_raw": recall_hit,
                       "recall_hit_after_filters": recall_hit_filtered,
                       "recall_hit_exact_title_excluded": recall_hit_notitle,
                       "dropped_exact_title": dropped_exact_title,
                       "hits": {f"acc_at_{k}": int(sum(v)) for k, v in stats.items()},
                       "hits_exact_title_excluded": {f"acc_at_{k}": int(sum(v))
                                                     for k, v in stats_notitle.items()}}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()