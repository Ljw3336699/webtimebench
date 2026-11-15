"""Agent action prototypes for temporal evidence acquisition (CPU-testable).

These are the extended action space beyond Search-R1's search/answer pair,
implemented as pure, dependency-free functions so they can be validated on CPU
and later wired into the GPU rollout environment:

  * typed_gap_query      -> <request_role>: build a query that asks for a
                            specific temporal role (publication/modification/event).
  * aggregate_evidence_dates -> <aggregate_dates>: combine dates across retrieved
                            evidence with rank/similarity weighting.
  * verify_date_candidates -> <verify_date>: confirm/refute candidate dates by
                            searching for each one (search_fn injectable).
  * annotate_date_roles   -> date-role tagger for evidence records (metadata +
                            content cues; roles: publish/modify/event/unknown).

Search_fn signature: f(query: str) -> list of evidence dicts with keys
url/title/content/publication_date (rank order = returned order).
"""
from __future__ import annotations

import collections
import datetime as dt
import math
import re

_STOP = {"the", "a", "an", "and", "or", "is", "are", "was", "were", "be", "to", "of",
         "for", "in", "on", "at", "by", "with", "from", "that", "this", "it", "as"}
_INTENTS = {
    "publication": ["publication date", "发布日期", "published date", "first published"],
    "modification": ["last update", "更新时间", "modified date", "latest version"],
    "event": ["event date", "事件日期", "occurred date", "took place"],
    "general": ["date", "日期"],
}


def typed_gap_query(body, intent="publication", n_entities=6):
    """Build a query conditioned on the missing temporal role (typed-gap query).

    Uses the leading salient tokens of the document plus a role-specific suffix.
    This implements the <request_role> action deterministically (template form);
    a learned version can replace the suffix generator later."""
    if intent not in _INTENTS:
        raise ValueError(f"Unknown intent: {intent}; choose from {sorted(_INTENTS)}")
    tokens = [t for t in re.findall(r"\S+", body) if t.lower() not in _STOP][:n_entities]
    suffix = " ".join(_INTENTS[intent][:2])
    return (" ".join(tokens) + " " + suffix).strip()


def _parse_date(value):
    if not isinstance(value, str):
        return None
    text = value.strip()[:10]
    try:
        return dt.date.fromisoformat(text)
    except ValueError:
        return None


def _content_dates(text, window=4000):
    """Extract plausible dates from text (shared with the diagnostics scripts)."""
    out = set()
    pats = [
        re.compile(r"(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})"),
        re.compile(r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"),
    ]
    for rx in pats:
        for m in rx.finditer(text[:window]):
            g = m.groups()
            try:
                d = dt.date(int(g[0]), int(g[1]), int(g[2]))
                if 2000 <= d.year <= dt.date.today().year:
                    out.add(d)
            except ValueError:
                continue
    return sorted(out)


def _rank_weight(rank, k=2.0):
    return 1.0 / math.log2(rank + k)


def aggregate_evidence_dates(records, alpha=0.05):
    """Aggregate dates across evidence records (<aggregate_dates> action).

    Each record may carry publication_date (metadata) and content dates. Returns
    the rank-weighted median of all candidate dates (deterministic), or None.
    Weights: rank 1/log2(r+2); metadata dates are weighted 2x content dates."""
    candidates = []
    for rank, rec in enumerate(records, 1):
        meta = _parse_date(rec.get("publication_date"))
        if meta:
            candidates.append((meta, 2.0 * _rank_weight(rank)))
        for d in _content_dates(rec.get("content") or rec.get("text") or ""):
            candidates.append((d, _rank_weight(rank)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    total = sum(w for _, w in candidates)
    acc = 0.0
    for d, w in candidates:
        acc += w
        if acc >= total / 2:
            return d.isoformat()
    return candidates[-1][0].isoformat()


def verify_date_candidates(candidates, search_fn, topk=3, tol_days=3):
    """Confirm/refute candidate dates by searching (<verify_date> action).

    For each candidate, searches "<entity context> <date>" and reports whether
    any returned evidence carries the candidate date (within tol_days) in its
    metadata or content. Returns list of {candidate, confirmed, evidence}."""
    if not candidates:
        return []
    results = []
    for cand in candidates:
        query = f"{cand} date"  # caller may extend with context tokens
        try:
            hits = search_fn(query, topk) if callable(search_fn) else []
        except Exception:
            hits = []
        confirmed = False
        best = None
        for rec in hits:
            meta = _parse_date(rec.get("publication_date"))
            cds = _content_dates(rec.get("content") or rec.get("text") or "")
            target = dt.date.fromisoformat(cand)
            for d in [meta] + cds:
                if d is None:
                    continue
                if abs((d - target).days) <= tol_days:
                    confirmed = True
                    best = d.isoformat()
                    break
            if confirmed:
                break
        results.append({"candidate": cand, "confirmed": confirmed,
                        "evidence_date": best, "query": query})
    return results


_ROLE_CUES = {
    "publish": re.compile(r"(publish|posted|发布|发表|首发|release[d]? on)", re.I),
    "modify": re.compile(r"(updated|last updat|modified|revised|更新|修订|最后更新)", re.I),
    "event": re.compile(r"(occurred|happened|was held|took place|announced|发生[于在]|举行|举办)", re.I),
}


def annotate_date_roles(records):
    """Tag each evidence record's date with a temporal role (<role tagger>).

    Uses metadata field names (publication/modified) and content cue words.
    Returns records with an added 'date_role' field per candidate date.
    Roles: publish / modify / event / unknown."""
    tagged = []
    for rec in records:
        roles = []
        meta = rec.get("source_date_field") or ""
        if "modif" in meta or "update" in meta:
            roles.append("modify")
        elif "publish" in meta or "date" in meta:
            roles.append("publish")
        text = (rec.get("content") or rec.get("text") or "")[:2000]
        for role, rx in _ROLE_CUES.items():
            if rx.search(text):
                roles.append(role)
        roles = list(dict.fromkeys(roles)) or ["unknown"]
        out = dict(rec)
        out["date_role"] = roles[0]
        tagged.append(out)
    return tagged


def run_self_test():
    """Quick CPU smoke over the four actions (no network)."""
    body = "NASA Artemis II mission update published 2025-01-01 by the agency."
    q = typed_gap_query(body, "publication")
    assert "publication date" in q, q
    records = [
        {"publication_date": "2025-01-01", "content": "Updated 2025-01-03 notes."},
        {"publication_date": None, "content": "Event took place 2025-01-01."},
    ]
    agg = aggregate_evidence_dates(records)
    assert agg is not None and agg.startswith("2025-01"), agg
    hits = verify_date_candidates(["2025-01-01"],
                                  lambda q, k=3: [{"publication_date": "2025-01-02", "content": ""}])
    assert hits[0]["confirmed"] is True, hits
    roles = annotate_date_roles([{"source_date_field": "modified", "content": "updated notes"}])
    assert roles[0]["date_role"] == "modify", roles
    print("agent_actions self-test OK:", q, "|", agg, "|", hits[0]["confirmed"], "|", roles[0]["date_role"])


if __name__ == "__main__":
    run_self_test()
