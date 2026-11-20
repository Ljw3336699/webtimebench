# WebTimeBench dataset card

## What is new vs reused

| Component | Status | Notes |
|---|---|---|
| WebTimeBench splits | **New** | Self-collected public web documents; cleaned, exact-normalized-body disjoint; `clean` and `temporal` variants. |
| EXP/IMP/DIS regime labels | **New** | Assigned by the project pipeline; 102-record author audit (96 confirmations, 6 proposed corrections). |
| Evaluation protocol | **New** | Full-denominator metrics, temporal-shift split, four-view design, optional retrieval date window. |
| Frozen BM25 indexes | **New build** | SQLite/FTS5, SHA-256 manifests; rebuildable from shipped corpora. |
| WCEP full / pilot | **Reused** | Upstream human-annotated event boundaries and article dates; article text retains upstream terms. |
| WebNewsEvents | **Re-processed** | Clustered from CC-News 2017–2018 by normalized title; gold dates are upstream article dates. |

## Records and fields

- Nominal collection: 23,000 public web documents with metadata stripped before modeling.
- Released: 21,374 records (`clean` 17,557/923/2,894; `temporal` 9,087/471/1,448).
- Fields: `id`, sanitized `prompt` body, one ISO `golden_answers` date, `difficulty`
  (EXP/IMP/DIS), `extra_info` (body hash, split, label status), `reward_model` target.
- Languages: multilingual web documents; dates normalized to ISO and validated as real
  calendar dates.
- Source URLs: **not recoverable** from the export. Source- and event-disjointness are not
  established; only leakage proxies are reported.

## Splits

- `clean`: exact-normalized-body disjoint; original test bodies are excluded from
  train/dev (497 duplicate bodies, 546 train bodies appearing in the original test set).
- `temporal`: train/dev targets ≤ 2023-12-31, test ≥ 2024-01-01; removes the modal-date
  prior (Acc@30 0.129 clean vs 0.000 shifted).

## Label audit

- 102 records stratified by regime (34 each), seed 20261116, sampled from the clean
  training split.
- Author review: 96 confirmations, 6 proposed regime corrections (1 IMP→DIS, 5 DIS→IMP).
- Labels were **not modified** in this release; reference dates were not independently
  re-verified against source pages. The audit is single-annotator and not an IAA study.

## Licenses

- Splits: CC BY 4.0. Code: Apache-2.0.
- Event corpora: upstream terms apply (see `LICENSE_NOTES.md`); article-text
  redistribution requires compliance with upstream licenses.

## Known limitations

- Missing source URLs and unverified source/event disjointness.
- WebNewsEvents clusters are automatic; human boundaries exist only in WCEP.
- Event evaluators exclude only the same corpus id; SimHash near-duplicate filtering is
  not implemented.
- BM25 is the default retriever; a multilingual E5 dense control is reported in
  Document-level retrieval baselines: see `reports/PROVENANCE.md`.

## Maintenance

SHA-256 manifests (`MANIFEST.sha256`), versioned GitHub releases, and rebuild scripts are
shipped. Report issues via the repository issue tracker.
