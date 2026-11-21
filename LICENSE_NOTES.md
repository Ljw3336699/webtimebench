# License and provenance notes

Each shipped resource has distinct origin and license terms. Reviewers and users must
not assume a single license applies.

| Resource | Origin | License/status | Redistribution note |
|---|---|---|---|
| WebTimeBench splits | Self-collected by this project; cleaned, split, and hash-pinned here | CC BY 4.0 (as declared in the paper) | Benchmark splits are derived artifacts; source-web-content rights remain with original publishers. |
| WebNewsEvents | Derived from HuggingFace `vblagoje/cc_news` (5 parquet shards, 2017-2018) | Follow the upstream dataset license on the HF dataset card; article text copyright remains with original publishers | We ship the *processed* article list with titles/text as used for the event corpus; users must comply with upstream terms. |
| WCEP full & pilot | Wikipedia Current Events Portal dataset (complementizer/wcep-mds-dataset, ACL 2020); editors define the events and cite sources; most articles were added from Common Crawl by an automatic event assignment | Upstream dataset terms apply; article text rights remain with their publishers | We ship event annotations (editor-provided dates) and article metadata/text extracted from the upstream release. |
| Time-Bench (not shipped here) | ulab-ai/Time-Bench (NYT Archive API-derived) | Upstream license applies | Only referenced as an external baseline resource; not redistributed. |
| Code (scripts/) | This project | Apache-2.0 (as declared in the paper; see `training_code/LICENSE`) | Free to use with attribution. |

## Important limitations

- Article *text* provenance and redistribution rights are not independently verified;
  users should re-check upstream terms before industrial use.
- The WCEP `publish_data`/`time` fields are automatically extracted and only 83.2% are
  parseable; the WCEP event date (editor-supplied) is the reliable gold label.
- WebNewsEvents clusters are automatic (normalized title + 45-day window); editor-defined
  events with automatically assigned articles are available only in the WCEP corpora.