
"""Frozen local retrieval using the Python standard library and SQLite FTS5.

The index contains source documents and their observed metadata, never benchmark
gold-label records. Query processing is lexical BM25, not semantic retrieval.
"""
from __future__ import annotations

import datetime as dt
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import unicodedata

from timeagent.evidence import canonical_record

INDEX_SCHEMA = "timeagent-local-fts5-v1"
TOKENIZER_VERSION = "nfkc-lower-word-han-unigram-bigram-v1"
MAX_QUERY_TERMS = 128
LABEL_FIELDS = {"gold", "golden_answers", "gold_date", "golds", "target",
                "reward_model", "prompt", "stop_prediction", "search_prediction"}


class BackendError(RuntimeError):
    """Retrieval failure, distinct from a successful empty result set."""


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_path(path):
    path = Path(path)
    return path.with_name(path.name + ".manifest.json")


def _han(character):
    point = ord(character)
    return 0x3400 <= point <= 0x9FFF or 0x20000 <= point <= 0x323AF


def tokenize(text):
    """Unicode words plus Han characters and adjacent Han bigrams.

    This small deterministic tokenizer needs no language model or dictionaries.
    Repetitions remain in documents for BM25 term frequency.
    """
    if not isinstance(text, str):
        raise ValueError("Text must be a string")
    text = unicodedata.normalize("NFKC", text).lower()
    terms, pending, kind = [], [], None

    def flush():
        if not pending:
            return
        value = "".join(pending)
        if kind == "han":
            terms.extend(value)
            terms.extend(value[i:i + 2] for i in range(len(value) - 1))
        else:
            terms.append(value)
        pending.clear()

    for ch in text:
        next_kind = "han" if _han(ch) else "word" if ch.isalnum() else None
        if next_kind != kind or next_kind is None:
            flush()
            kind = next_kind
        if next_kind is not None:
            pending.append(ch)
    flush()
    return terms


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def normalize_corpus_record(row):
    if not isinstance(row, dict):
        raise ValueError("Each corpus row must be an object")
    if LABEL_FIELDS & row.keys():
        raise ValueError("Corpus rows must be source evidence, not benchmark gold/prompt records")
    identity = row.get("corpus_id", row.get("id"))
    if not isinstance(identity, str) or not identity.strip() or identity != identity.strip():
        raise ValueError("Each corpus row needs a nonempty corpus_id (or id)")
    if "id" in row and "corpus_id" in row and row["id"] != row["corpus_id"]:
        raise ValueError("id and corpus_id disagree")
    for key in ("title", "content", "snippet", "url"):
        if row.get(key) is not None and not isinstance(row[key], str):
            raise ValueError(f"{identity}: {key} must be a string or null")
    title = row.get("title") or ""
    content = row.get("content") or row.get("snippet") or ""
    if not content.strip() or not tokenize(content):
        raise ValueError(f"{identity}: a nonempty source body is required")
    date = row.get("publication_date")
    if date is not None:
        try:
            if not isinstance(date, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", date):
                raise ValueError
            dt.date.fromisoformat(date)
        except ValueError:
            raise ValueError(f"{identity}: publication_date must be a real YYYY-MM-DD date or null") from None
    canonical = canonical_record({"url": row.get("url") or "", "title": title,
                                  "content": content, "snippet": row.get("snippet") or "",
                                  "publication_date": date}, 0)
    canonical["corpus_id"] = identity
    canonical["source_date_evidence"] = row.get("source_date_evidence")
    # Preserve asserted provenance without upgrading it to verified ground truth.
    for key in ("snapshot_sha256", "source_date_field", "source_date_raw", "event_id",
                "content_sha256", "body_sha256", "date_label_status", "fetched_at_utc",
                "retrieved_at_utc", "snapshot_path", "synthetic", "merged_sources",
                "discovery_inputs", "relevance_review_status", "needs_date_review", "date_review_reason"):
        if key in row:
            canonical[key] = row[key]
    _json(canonical)  # Reject non-finite/unserializable audit metadata.
    return canonical


def build_index(corpus, output):
    corpus, output = Path(corpus).resolve(), Path(output).resolve()
    sidecar = manifest_path(output)
    if output.exists() or sidecar.exists():
        raise ValueError("Index or manifest already exists; choose a new versioned path")
    if not corpus.is_file():
        raise ValueError("Corpus JSONL file does not exist")
    source_hash = sha256_file(corpus)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".timeagent-index-", suffix=".sqlite", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary)
    count, identities, record_hash = 0, set(), hashlib.sha256()
    try:
        with closing(sqlite3.connect(temporary)) as db, db:
            db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("CREATE TABLE documents (rowid INTEGER PRIMARY KEY, corpus_id TEXT UNIQUE NOT NULL, record TEXT NOT NULL)")
            try:
                db.execute("CREATE VIRTUAL TABLE corpus_fts USING fts5(title_terms, body_terms, content='', tokenize='unicode61 remove_diacritics 2')")
            except sqlite3.OperationalError as error:
                raise ValueError("This Python SQLite build lacks FTS5; use a Python build with FTS5 enabled") from error
            with corpus.open(encoding="utf-8") as stream:
                for line_no, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        row = normalize_corpus_record(json.loads(line))
                    except (ValueError, TypeError) as error:
                        raise ValueError(f"Corpus line {line_no}: {error}") from error
                    identity = row["corpus_id"]
                    if identity in identities:
                        raise ValueError(f"Duplicate corpus_id: {identity}")
                    identities.add(identity)
                    count += 1
                    encoded = _json(row)
                    record_hash.update((encoded + "\n").encode())
                    db.execute("INSERT INTO documents VALUES (?, ?, ?)", (count, identity, encoded))
                    db.execute("INSERT INTO corpus_fts(rowid, title_terms, body_terms) VALUES (?, ?, ?)",
                               (count, " ".join(tokenize(row["title"])), " ".join(tokenize(row["content"]))))
            if not count:
                raise ValueError("Corpus must contain at least one source document")
            if source_hash != sha256_file(corpus):
                raise ValueError("Corpus changed while the index was being built")
            metadata = {"schema": INDEX_SCHEMA, "tokenizer_version": TOKENIZER_VERSION,
                        "engine": "sqlite-fts5-bm25", "sqlite_version": sqlite3.sqlite_version,
                        "source_corpus_sha256": source_hash, "document_count": count,
                        "canonical_records_sha256": record_hash.hexdigest(),
                        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "ranking": "bm25(title_terms weight=4, body_terms weight=1), corpus_id tie-break",
                        "query_terms": "OR over unique literal tokenizer terms, maximum 128"}
            db.executemany("INSERT INTO metadata VALUES (?, ?)", [(k, _json(v)) for k, v in metadata.items()])
            db.execute("INSERT INTO corpus_fts(corpus_fts) VALUES ('optimize')")
        temporary.replace(output)
        manifest = {**metadata, "source_corpus_path": str(corpus), "index_sha256": sha256_file(output)}
        sidecar.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return manifest
    finally:
        if temporary.exists():
            temporary.unlink()


class LocalCorpusBackend:
    """Read-only, version-checked local index compatible with search snapshots."""
    def __init__(self, index):
        self.path = Path(index).resolve()
        self.manifest = json.loads(manifest_path(self.path).read_text(encoding="utf-8"))
        if self.manifest.get("schema") != INDEX_SCHEMA or self.manifest.get("tokenizer_version") != TOKENIZER_VERSION:
            raise ValueError("Unsupported index schema or tokenizer version")
        actual_hash = sha256_file(self.path)
        if actual_hash != self.manifest.get("index_sha256"):
            raise ValueError("Index SHA-256 does not match its manifest")
        self.backend_id = "local-sqlite-fts5-sha256:" + actual_hash
        self._stat = self._file_signature()
        with closing(self._connect()) as db:
            meta = {k: json.loads(v) for k, v in db.execute("SELECT key, value FROM metadata")}
            if any(meta.get(k) != self.manifest.get(k) for k in meta):
                raise ValueError("Index metadata differs from its sidecar manifest")
            if meta.get("schema") != INDEX_SCHEMA or meta.get("tokenizer_version") != TOKENIZER_VERSION:
                raise ValueError("Unsupported internal index metadata")
            if db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] != self.manifest.get("document_count"):
                raise ValueError("Index document count mismatch")
            # Force FTS availability/schema validation before serving a request.
            db.execute("SELECT rowid FROM corpus_fts LIMIT 1").fetchone()

    def _file_signature(self):
        stat = self.path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def _connect(self):
        # Each worker opens its own immutable read-only connection.
        db = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        db.execute("PRAGMA query_only=ON")
        return db

    def search(self, query, topk, max_date=None):
        if type(topk) is not int or not 1 <= topk <= 100:
            raise ValueError("topk must be an integer in [1,100]")
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms:
            raise ValueError("Query has no searchable tokens")
        if len(terms) > MAX_QUERY_TERMS:
            raise ValueError(f"Query exceeds the {MAX_QUERY_TERMS}-term local limit")
        expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        where = "corpus_fts MATCH ?"
        params = [expression]
        if max_date is not None:
            # Temporal leakage gate: drop evidence dated after the target window;
            # null dates (unverified) are kept so evidence without a date still retrieves.
            where += (" AND (json_extract(documents.record, '$.publication_date') IS NULL"
                      " OR json_extract(documents.record, '$.publication_date') <= ?)")
            params.append(max_date)
        params.append(topk)
        try:
            if self._file_signature() != self._stat:
                raise BackendError("Frozen local index changed after startup")
            with closing(self._connect()) as db:
                matches = db.execute(
                    "SELECT documents.record, bm25(corpus_fts, 4.0, 1.0) AS distance "
                    "FROM corpus_fts JOIN documents ON documents.rowid=corpus_fts.rowid "
                    f"WHERE {where} ORDER BY distance, documents.corpus_id LIMIT ?",
                    params).fetchall()
        except (OSError, sqlite3.DatabaseError) as error:
            raise BackendError(f"Local index read failed ({type(error).__name__})") from None
        results = []
        for rank, (encoded, distance) in enumerate(matches, 1):
            row = json.loads(encoded)
            row["rank"] = rank
            row["retrieval_score"] = -distance
            row["retrieval_score_definition"] = "negative SQLite FTS5 bm25 distance; larger is better"
            results.append(row)
        return results