# WebTimeBench Release (v1, 2026-11-15)

Resources for temporal localization of web documents: a document-dating benchmark with
temporal-shift and evidence-regime splits, frozen retrieval indexes (reproducible from the
shipped corpora), an evaluation protocol (full-denominator metrics, four-view ablation,
retrieval time-window filter), and event-level corpora for when retrieval *does* help.

## Scope and positioning

WebTimeBench is a dataset-and-evaluation resource for agentic information retrieval. Its
primary contribution is a reusable testbed for retrieval timing, evidence quality, and
cost-sensitive search decisions in temporal localization; the PPO policy is a usage example,
not the resource contribution.

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

Large corpora under `events/` and the `*.parquet`/`*.jsonl` dataset files are stored
with Git LFS. Run `git lfs pull` before the commands above, or rebuild the tabular
splits locally with `scripts/build_dataset.py` (the JSON splits are plain files).

## Key results reproduced by these artifacts

Event-level results run from the shipped corpora (samples are systematic, every 128th article):

- WebNewsEvents (automatic title clusters), title query, top-3: Acc@30 0.841 and same-event
  recall 0.978 as shipped; **0.216 and 0.005** after exact-title de-duplication (body-field
  queries: 0.765 -> 0.283). Random retrieval: 0.091.
- WCEP (human-annotated events): Acc@30 0.505 and recall 0.748 as shipped; **0.379 and 0.555**
  after de-duplication. Random retrieval: 0.084.
- Corpus-level audit (`scripts/audit_event_duplication.py`): fraction of same-event pairs with
  identical normalized titles is 0.996 (WebNewsEvents), 0.058 (our WCEP processing), 0.042
  (WCEP as published). `scripts/audit_split_overlap.py` reports within-split duplication and
  held-out items already in training for other resources.

Document-level retrieval measurements in the paper were taken on a crawled evidence pool that
cannot be redistributed; they are not reproducible from this release (see
`reports/PROVENANCE.md`).

## Training code

`training_code/` contains the modified Search-R1/verl fork used to train the policy
(Qwen2.5-7B-Instruct; PPO; 400 steps; uniform per-search cost, `--search-cost`;
details in the paper's setup section). It includes the patched `verl` library, the
`timeagent` task package, `pyproject.toml`/`requirements.txt`, and the training entry
scripts. The base model uses open Qwen2.5-7B-Instruct weights; trained checkpoints are
not shipped due to size and can be requested via repository issues.

```bash
pip install -r training_code/requirements.txt
# training (see paper for exact hyperparameters):
python scripts/launch.py train --model <qwen2.5-7b-instruct> --variant temporal \
    --gpus 8 --steps 400 \
    --quality-scales-days 7 30 365 --quality-weights 0.4 0.4 0.2 --invalid-penalty 0.3 \
    --search-cost 0.1 --test-freq 25 --save-freq 100 --keep-checkpoints 3 \
    --retriever-url <frozen-index-server-url> --code-root training_code --run
```

`--keep-checkpoints N` keeps only the N newest actor checkpoints and deletes critic
states (only needed for resuming), which bounds disk use on shared machines.

## Citation

(to be added)

## License

See `LICENSE_NOTES.md` for per-source licensing and redistribution notes.
## Known limitations

- Source URLs are not recoverable for the document-dating split, so source-disjointness is not
  claimed; the split is released as derived annotations with a hash-pinned rebuild path.
- De-duplication is exact-title only; semantic near-duplicates are not removed, so the
  de-duplicated event-level numbers remain upper bounds.
- `max_date` is an optional serving mode and was not enabled in the training runs.
- Trained-policy runs are unstable late in training; checkpoints must be selected on dev.
- `--cost-by-regime` was silently ignored in releases up to 13a6795 (the reward received only the
  date, so every regime was charged the default cost); it is fixed and covered by
  `tests/test_reward.py`. All reported runs used a uniform per-search cost.

## Retrieval environment API

`scripts/serve_search.py` exposes four endpoints over HTTP on `--bind`/`--port`.

- `POST /retrieve` — body `{"queries": [str, ...], "topk": int, "max_date": "YYYY-MM-DD"|null}`.
  Returns `{"result": [[hit, ...], ...]}` aligned with `queries`. Each hit carries
  `document.contents` (first line is the publication date or `withheld`/`unknown`),
  `rank`, `view`, `publication_date_used`, and `provenance` with the full canonical record.
  A query with no searchable tokens, a `topk` outside `[1,100]` or an oversized body is a
  4xx; a backend failure is a 5xx and is recorded in the snapshot.
- `GET /healthz` — liveness.
- `GET /readyz` — readiness plus `backend_id`, `document_count` and `index_sha256`.
- `GET /stats` — request counters for the running instance.

Serving options that define an experimental condition:

| option | effect |
|---|---|
| `--view full\|text\|dates\|shuffled-dates` | evidence shown to the agent |
| `--max-date YYYY-MM-DD` | drops dated evidence after the window; null dates kept |
| `--retrieval bm25\|random` | random serves unrelated documents at matched budget |
| `--replay-snapshot FILE` | answers only from a recorded snapshot; no network |
| `TIMEAGENT_NULL_RETRIEVER=1` | keeps the agent loop, returns no evidence |
| `--reward naiverag` (launcher) | forced retrieval, one call per document |

Every served request appends one `timeagent-search-snapshot-v1` record carrying the query,
`topk`, `backend` identity, `view`, `seed` and `max_date`, which is what makes a run replayable.
Control backends report a distinct `backend_id` (`local-sqlite-random-control-sha256:...`), so a
control can never be mistaken for a live run.

## Verifying the release

```bash
bash examples/replay_equivalence.sh                                  # replay equivalence demo
PYTHONPATH=src python3 -m unittest discover -s tests -v              # 17 environment tests
python3 scripts/audit_event_duplication.py \
    --input events/news_events/articles.jsonl --event-key event_id \
    --title-key title --body-key text --date-key date                # near-duplicate audit
```

`scripts/audit_event_duplication.py` reports the fraction of same-event article pairs that
share a normalized title. Run it before reporting any event-level retrieval number: 0.996 on
WebNewsEvents accompanies a 3.9-fold accuracy drop under de-duplication, 0.058 on WCEP a
1.3-fold drop.

## Auditing other collections

`scripts/audit_split_overlap.py` measures exact-normalized duplication within a split and
held-out items already present in training, with no index. Readers exist for Time-Bench style
prompts, WCEP's nested event files, title/article pairs and WebTimeBench splits:

```bash
python3 scripts/audit_split_overlap.py --schema wcep --label WCEP \
    --train source/wcep/train.jsonl.gz --test source/wcep/val.jsonl.gz source/wcep/test.jsonl.gz
```

Results for the resources audited in the paper are in `reports/split_audit/summary.json`.
