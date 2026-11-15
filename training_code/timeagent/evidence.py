"""Pure, deterministic views of one retrieved evidence pool."""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
import random
import re

VIEWS = ("full", "text", "dates", "shuffled-dates")


def normalize_publication_date(value):
    """Convert explicit dates or positive Unix seconds/milliseconds in UTC.

    Zero, negative and nonfinite timestamps are unknown, not 1970 defaults.
    Datetime strings must include a zone; naive local times are not guessed.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        if value != value.strip() or not value:
            return None
        if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
            try:
                return dt.date.fromisoformat(value).isoformat()
            except ValueError:
                return None
        if "T" in value:
            try:
                parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    return None
                return parsed.astimezone(dt.timezone.utc).date().isoformat()
            except ValueError:
                return None
    try:
        timestamp = float(value)
        if not math.isfinite(timestamp) or timestamp <= 0:
            return None
        if timestamp > 10_000_000_000:
            timestamp /= 1000
        return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).date().isoformat()
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def _text(value):
    return value if isinstance(value, str) else "" if value is None else str(value)


def canonical_record(item, rank):
    if not isinstance(item, dict):
        raise ValueError("Each backend result must be an object")
    field = next((key for key in ("page_time", "publication_date", "pub_time", "date") if key in item), None)
    raw = item.get(field) if field else None
    if isinstance(raw, float) and not math.isfinite(raw):
        raw = str(raw)
    return {"rank": rank, "url": _text(item.get("url", item.get("link"))),
            "title": _text(item.get("title")), "snippet": _text(item.get("snippet")),
            "content": _text(item.get("content")), "source_date_field": field,
            "source_date_raw": raw, "publication_date": normalize_publication_date(raw)}


def pool_fingerprint(records):
    text = json.dumps(records, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def evidence_view(records, view="full", seed=20261103, query=""):
    if view not in VIEWS:
        raise ValueError(f"Unknown evidence view {view!r}")
    if type(seed) is not int:
        raise ValueError("The shuffle seed must be an integer")
    dates = [row["publication_date"] for row in records]
    if view == "shuffled-dates":
        indices = [i for i, date in enumerate(dates) if date is not None]
        available = [dates[i] for i in indices]
        local_seed = hashlib.sha256(f"{seed}\n{query}\n{pool_fingerprint(records)}".encode()).digest()
        random.Random(int.from_bytes(local_seed, "big")).shuffle(available)
        for index, date in zip(indices, available):
            dates[index] = date
    output = []
    for row, date in zip(records, dates):
        # generation._passages2string treats line 1 as publication time.
        # Only document.contents is model-visible in that adapter.
        first_line = "withheld" if view == "text" else date or "unknown"
        text = row["content"] or row["snippet"]
        visible = [] if view == "dates" else (["Title: " + row["title"]] if row["title"] else []) + ([text] if text else [])
        output.append({"document": {"contents": first_line + "\n" + "\n".join(visible)},
                       "rank": row["rank"], "view": view,
                       "publication_date_used": None if view == "text" else date,
                       "provenance": copy.deepcopy(row)})
    return output
