# Provenance of reported values

Each value in the paper, the script that produces it, its inputs, and whether it can be
re-run from this release.

## Reproducible from the release

| Value | Script | Inputs |
|---|---|---|
| WebNewsEvents Acc@30 0.841 / 0.216, recall 0.978 / 0.005 (n = 10,072 sample) | `scripts/evaluate_wcep_aggregation.py --shard i --num-shards 128` | `events/news_events/{articles,corpus}.jsonl`, index built by `scripts/build_local_index.py` |
| WebNewsEvents body query 0.765 / 0.283 | same, `--query-field body` | same |
| WCEP Acc@30 0.505 / 0.379, recall 0.748 / 0.555 (n = 7,251 sample) | `scripts/evaluate_wcep_full_article.py --shard i --num-shards 128` | `events/wcep_full/corpus.jsonl` |
| Random-retrieval floors 0.091 / 0.084 | same scripts, `--retrieval random` | same corpora |
| WCEP held-out events (val+test, 2,042 events): Acc@30 0.605 / 0.410, recall 0.771 / 0.534, random 0.204 (n = 7,207 sample) | `scripts/prepare_wcep_full.py --splits val.jsonl.gz test.jsonl.gz`, then `scripts/build_local_index.py` and `scripts/evaluate_wcep_full_article.py --shard 0 --num-shards 4` (add `--exclude-exact-title` / `--retrieval random`) | upstream WCEP val and test files; results in `reports/wcep_heldout/` |
| Same-title pair rates 0.996 / 0.058 / 0.042 | `scripts/audit_event_duplication.py` | `events/news_events/articles.jsonl`, `events/wcep_full/corpus.jsonl`, upstream WCEP `train.jsonl.gz` |
| Split audit table (WCEP, CC-News pairs, Time-Bench, WebTimeBench) | `scripts/audit_split_overlap.py` | upstream public releases; results in `reports/split_audit/` |
| Matched month baseline 0.061 | computed from the gold year-month distribution of `webtimebench/temporal/test.json` | released split |
| Replay equivalence, 17 environment tests | `examples/replay_equivalence.sh`, `tests/test_environment.py` | `events/wcep_pilot/corpus.jsonl` |

Event-level numbers are on systematic samples (every 128th article) because a full BM25 pass
over 161,122 articles takes about 13 hours on one process; shards merge exactly by summing the
`sums` block of each output.

## Measured on a pool that cannot be redistributed (not re-runnable from the release)

Document-level evidence-quality and oracle values (oracle Acc@30 0.164 unfiltered and 0.294
with `max_date=2025-12-31`, dense E5 0.130, month-overlap rates) and the trained-policy test
scores were measured against a crawled evidence pool and local checkpoints that are not part of
this release. They are reported in the paper as auxiliary evidence; no claim of the protocol
depends on them.
