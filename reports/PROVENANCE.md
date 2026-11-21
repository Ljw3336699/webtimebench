# Provenance of reported values

Each value in the paper, the script that produces it, its inputs, and whether it can be
re-run from this release.

## Reproducible from the release

| Value | Script | Inputs |
|---|---|---|
| WebNewsEvents Acc@30 0.841 / 0.216, recall 0.978 / 0.005 (n = 10,072 sample) | `scripts/evaluate_wcep_aggregation.py --shard i --num-shards 128` | `events/news_events/{articles,corpus}.jsonl`, index built by `scripts/build_local_index.py` |
| WebNewsEvents body query 0.765 / 0.283 | same, `--query-field body` | same |
| WCEP Acc@30 0.505 / 0.379, recall 0.748 / 0.555 (n = 7,251 sample) | `scripts/evaluate_wcep_full_article.py --shard i --num-shards 128` | `events/wcep_full/corpus.jsonl` |
| Random-retrieval floors 0.088 / 0.076 on the same 10,072 / 7,251 queries (shards 0-7 of 128) | same scripts, `--retrieval random --shard i --num-shards 128` | `reports/random_same_sample/random_floor_same_sample.json` |
| WCEP held-out events (val+test, 2,042 events): Acc@30 0.605 / 0.410, recall 0.771 / 0.534, random 0.204 (n = 7,207 sample) | `scripts/prepare_wcep_full.py --splits val.jsonl.gz test.jsonl.gz`, then `scripts/build_local_index.py` and `scripts/evaluate_wcep_full_article.py --shard 0 --num-shards 4` (add `--exclude-exact-title` / `--retrieval random`) | upstream WCEP val and test files; results in `reports/wcep_heldout/` |
| WCEP restricted to editor-cited or assignment >= 0.99 articles (76,878): Acc@30 0.521 / 0.395, recall 0.814 / 0.628 (n = 4,030) | `scripts/filter_wcep_by_assignment.py`, then `scripts/build_local_index.py` and `scripts/evaluate_wcep_full_article.py --shard 0 --num-shards 16` | upstream WCEP train file; results in `reports/wcep_confident/` |
| Untrained-model table: Acc@30 under none / real / random evidence, six models, n = 1,448 (values on real and random use the non-redistributable evidence pool) | `scripts/launch.py evaluate` (verl evaluator; Qwen3-8B: `scripts/evaluate_standalone_agent.py`), table by `scripts/summarize_sweep.py` | `reports/sweep_summary.json` |
| Paired differences real-null, real-random and trained-minus-untrained (Acc@30, 2,000 draws) | `scripts/bootstrap_sweep_comparisons.py` | `reports/sweep_bootstrap.json` |
| Same-title pair rates 0.996 / 0.058 / 0.042 | `scripts/audit_event_duplication.py` | `events/news_events/articles.jsonl`, `events/wcep_full/corpus.jsonl`, upstream WCEP `train.jsonl.gz` |
| Split audit table (WCEP, CC-News pairs, Time-Bench, WebTimeBench) | `scripts/audit_split_overlap.py` | upstream public releases; results in `reports/split_audit/` |
| Single-draw year-month collision rate 0.061 (not the matched baseline for the any-of-5 criterion) | `scripts/evaluate_document_retrieval_baselines.py` | released split |
| Same-budget random-retrieval expectation for the any-of-5 year-month hit: 4.99% / 6.94% / 14.46% (non-redistributable pools) | `scripts/matched_random_baseline.py` | `reports/matched_random/*.json` |
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
