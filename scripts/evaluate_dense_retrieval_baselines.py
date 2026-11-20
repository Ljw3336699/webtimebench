#!/usr/bin/env python3
"""Dense-retrieval control for document dating (CPU, no GPU training needed).

Embeds a frozen evidence corpus and the temporal test queries with a
multilingual E5 model, then reports the same oracle/median/random controls as
the BM25 baseline script.
"""
import argparse
import datetime
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_document_retrieval_baselines import (  # noqa: E402
    median_date, parse_day, query_of, summarize,
)


def load_corpus(path):
    docs, dates, ids = [], [], []
    for line in Path(path).open(encoding="utf-8"):
        row = json.loads(line)
        text = (row.get("title") or "") + "\n" + (row.get("content") or "")
        docs.append("passage: " + text[:4000])
        dates.append(parse_day(row.get("publication_date")))
        ids.append(row.get("corpus_id"))
    return docs, dates, ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=ROOT / "webtimebench/temporal/test.json")
    parser.add_argument("--model", default=str(ROOT.parent / "model/embeddings/multilingual-e5-small"))
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    from sentence_transformers import SentenceTransformer

    docs, doc_dates, doc_ids = load_corpus(args.corpus)
    rows = json.loads(args.split.read_text(encoding="utf-8"))
    golds = [parse_day(r["golden_answers"][0]) for r in rows]
    queries = ["query: " + query_of(r) for r in rows]

    model = SentenceTransformer(args.model, device="cpu")
    doc_emb = model.encode(docs, batch_size=args.batch_size, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True)
    query_emb = model.encode(queries, batch_size=args.batch_size, normalize_embeddings=True,
                             show_progress_bar=True, convert_to_numpy=True)

    retrieved = []
    for start in range(0, len(rows), 64):
        block = query_emb[start:start + 64] @ doc_emb.T
        for scores in block:
            order = np.argpartition(-scores, args.topk)[:args.topk]
            order = order[np.argsort(-scores[order])]
            retrieved.append([doc_dates[i] for i in order if doc_dates[i] is not None])

    month_total = month_hit = same_day = 0
    for dates, gold in zip(retrieved, golds):
        if not dates:
            continue
        month_total += 1
        month_hit += any((d.year, d.month) == (gold.year, gold.month) for d in dates)
        same_day += any(d == gold for d in dates)

    pool = [d for dates in retrieved for d in dates]
    rng = random.Random(20261103)
    random_preds = [median_date([rng.choice(pool) for _ in range(5)]) if pool else None for _ in rows]

    baselines = {"oracle_best_retrieved": summarize(
        [min(dates, key=lambda d: abs((d - gold).days)) if dates else None
         for dates, gold in zip(retrieved, golds)], golds)}
    for k in range(1, args.topk + 1):
        baselines[f"median_top{k}"] = summarize(
            [median_date(dates[:k]) for dates in retrieved], golds)
    baselines["random_retrieved_date"] = summarize(random_preds, golds)

    report = {
        "schema": "timebench.dense_retrieval_baselines.v1",
        "corpus": str(args.corpus.resolve()),
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "split": str(args.split),
        "model": args.model,
        "n": len(rows),
        "topk": args.topk,
        "evidence_quality": {
            "queries_with_any_dated_hit": month_total,
            "gold_month_shared_by_any_hit": month_hit,
            "gold_month_rate": month_hit / month_total if month_total else None,
            "random_month_rate": 1 / 12,
            "gold_day_shared_by_any_hit": same_day,
        },
        "baselines": baselines,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["evidence_quality"], indent=2))
    for name in ("oracle_best_retrieved", "median_top3", "random_retrieved_date"):
        print(name, json.dumps(baselines[name], indent=2))


if __name__ == "__main__":
    main()
