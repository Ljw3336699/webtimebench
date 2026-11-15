# WebTimeBench Release (v1, 2026-11-15)

Resources for temporal localization of web documents: a document-dating benchmark with
temporal-shift and evidence-regime splits, frozen retrieval indexes (reproducible from the
shipped corpora), an evaluation protocol (full-denominator metrics, four-view ablation,
retrieval time-window filter), and event-level corpora for when retrieval *does* help.

## Contents

| Directory | Contents | Size |
|---|---|---|
| `webtimebench/` | `clean/{train,dev,test}.json/{.parquet}` (17,557/923/2,894), `temporal/{...}` (9,087/471/1,448, train/dev <= 2023, test >= 2024), `quarantine/`, `excluded/`, `id_mapping.json`, `manifest.json` | ~194 MB |
| `events/news_events/` | WebNewsEvents: 49,648 events / 161,122 articles (CC-News 2017-2018, normalized-title clustering, gold = article dates) + corpus + index manifest | ~1.7 GB |
| `events/wcep_full/` | WCEP (train split processed): 8,158 human-annotated events / 138,432 articles (83.2% with parsed dates) + corpus + index manifest | ~1.2 GB |
| `events/wcep_pilot/` | WCEP pilot: 50 events / 199 articles | ~2 MB |
| `scripts/` | dataset builder, corpus preparation, frozen local BM25 index builder, retrieval server (four views + time-window filter), WCEP/news evaluation, Time-Bench evaluation, PPO launcher, gate-experiment runner | |
| `reports/` | key experiment reports (JSON) | |

## Quick reproducibility

```bash
# 1. Build the frozen local BM25 index from a shipped corpus (SQLite FTS5, sha256-pinned):
python3 scripts/build_local_index.py --corpus events/news_events/corpus.jsonl \
    --output news_events_index.sqlite

# 2. Serve retrieval (four views supported: full/text/dates/shuffled-dates):
python3 scripts/serve_search.py --backend local --index news_events_index.sqlite \
    --view full --snapshot snap.jsonl

# 3. Evaluate event-level aggregation (full denominator, Acc@0/7/30 + same-event recall):
python3 scripts/evaluate_wcep_aggregation.py --articles events/news_events/articles.jsonl \
    --index news_events_index.sqlite --topk 3 --output reports/news_events_topk3.json

# 4. Reproduce WebTimeBench splits (source data hash-pinned): see scripts/build_dataset.py
```

## Key results reproduced by these artifacts

- Document-level retrieval ceiling (WebTimeBench temporal test): oracle **Acc@30 <= 0.16**;
  retrieved dates share the gold month 2.1% (below 8.3% random).
- Trained policy (3 seeds, regime-aware reward) learns to stop searching (54% -> 0%);
  controlled four-view ablation is exactly flat (Score 0.5552-0.5555).
- Event-level aggregation: WebNewsEvents **Acc@30 0.838** (top-3, recall 97.9%),
  WCEP full 0.598, WCEP pilot 0.598 - 3.7--5x above the document-level ceiling.

## Citation

(to be added)

## License

See `LICENSE_NOTES.md` for per-source licensing and redistribution notes.