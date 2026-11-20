# WebTimeBench evaluation protocol: one-page checklist

Use this list when reporting a result under the protocol. Each item names the script or file that implements it.

## Inputs
- Records: `id`, text, one ISO reference date (supplied with the data, not independently checked against the source page), regime label EXP/IMP/DIS (supplied; never shown to the model). Splits: `webtimebench/temporal/{train,dev,test}` (train and dev up to 2023, test 2024 onward).
- Retrieval corpus: a frozen SQLite FTS5 BM25 index. Report its SHA-256 from `index.sqlite.manifest.json`. A query window (`max_date`) must be stated if used.
- Evidence view: one of `full`, `text`, `dates`, `shuffled-dates` (`scripts/serve_search.py --view`).

## Report every rate with its denominator
- Document-level metrics: exact date, Acc@7, Acc@30 over all eligible records; an invalid or missing prediction counts as a failure (`src/timeagent/evaluation.py`).
- MAE and error percentiles only over valid predictions, always next to the prediction coverage.
- Retrieval hit rates: state both the rate over all queries and over queries that returned dated evidence.
- Compare any month-level hit rate with the year-month collision rate of the gold dates (0.0606 on the evidence set), not with 1/12.

## Budget and decoding
- Search budget: at most 2 executed searches per record, top-3 documents per search, plus one forced final turn. A search counts when it is dispatched, including failed requests.
- Decoding for evaluation: greedy. Training and evaluation use the same prompt template.
- Controls to run next to any retrieval result: null retriever (`TIMEAGENT_NULL_RETRIEVER=1`), random retrieval (`--retrieval random`, seed 20261119), and, for event-level tasks, exact-title de-duplication and body-field queries (`scripts/evaluate_wcep_aggregation.py`).

## Leakage controls and their scope
- Exact-title de-duplication under the clustering's own normalization. It does not remove semantic near-duplicates, so de-duplicated event-level numbers are upper bounds.
- Before any retrieval: same-event pair rates for identical normalized titles and bodies (`scripts/audit_event_duplication.py`) and cross-split overlap (`scripts/audit_split_overlap.py`).

## Uncertainty and seeds
- Wilson intervals for hit rates; paired bootstrap for model comparisons (2,000 resamples, seed 20261103). Report seed ranges separately from test-example intervals.
- Training results: state seed, step, and whether the checkpoint was selected on dev.

## Outputs to publish
- Prediction file with one row per id (`prediction`, `executed_searches`, `failed_searches`), merged metrics JSON, the retrieval snapshot, and the index digest.
- Offline replay check: `examples/replay_equivalence.sh` must reproduce identical evidence from the snapshot with no index and no network.

## Failure handling
- Retrieval timeout or HTTP error: no retry; the observation says that no evidence was returned and the attempt counts as a search.
- Empty or malformed answer: counted as invalid, not dropped.
- Rows with a missing id, an unknown id, or a duplicate id make the evaluator stop rather than score a partial set.
