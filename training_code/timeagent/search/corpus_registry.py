"""Build, verify and activate immutable source-corpus versions.

All writes are local. This module never collects URLs or turns target labels into
evidence. Readers pin the resolved index path for the lifetime of their process.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata
import urllib.parse
import uuid

from .local_backend import LocalCorpusBackend, build_index, normalize_corpus_record, sha256_file

VERSION_SCHEMA = "timeagent-corpus-version-v1"
CURRENT_SCHEMA = "timeagent-corpus-current-v1"
FILES = ("corpus.jsonl", "index.sqlite", "index.sqlite.manifest.json", "ingestion_report.json")


class RegistryBuildError(ValueError):
    def __init__(self, message, report_path):
        super().__init__(message + "; failure report: " + str(report_path))
        self.report_path = Path(report_path)


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _write_json(path, value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with Path(path).open("w", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def body_hash(content):
    value = " ".join(unicodedata.normalize("NFKC", content).lower().split())
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_url(value):
    if value in (None, ""):
        return ""
    if not isinstance(value, str) or value != value.strip():
        raise ValueError("Source URL must be a string without surrounding whitespace")
    parsed = urllib.parse.urlsplit(value)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username or parsed.password):
        raise ValueError("Source URL must use HTTP(S), without credentials")
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("Source URL has an invalid port") from None
    host = parsed.hostname.lower()
    if ":" in host:
        host = "[" + host + "]"
    if port is not None and not (parsed.scheme == "http" and port == 80 or parsed.scheme == "https" and port == 443):
        host += ":" + str(port)
    return urllib.parse.urlunsplit((parsed.scheme.lower(), host, parsed.path or "/", parsed.query, ""))


def _version(value):
    if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", value)
            or value.lower() == "current"):
        raise ValueError("Version must be 1..64 letters/digits/dot/underscore/hyphen, start with a letter or digit, and not be 'current'")
    return value


def _directory(registry, version):
    registry = Path(registry).resolve()
    directory = registry / "versions" / _version(version)
    if directory.resolve().parent != (registry / "versions").resolve():
        raise ValueError("Version directory escapes the registry")
    if (registry / "versions").is_symlink() or directory.is_symlink():
        raise ValueError("Registry version directories cannot be symbolic links")
    return directory


@contextmanager
def _writer_lock(registry):
    registry = Path(registry).resolve()
    registry.mkdir(parents=True, exist_ok=True)
    with (registry / ".write.lock").open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield registry
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSONL rows must be objects")
                yield line_number, row


def _target_exclusions(dataset_dirs):
    ids, urls, bodies, audit = set(), set(), set(), []
    for source in dataset_dirs:
        directory = Path(source).resolve()
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        dataset_audit = {"path": str(directory), "files": {}, "targets": 0}
        if manifest is not None:
            dataset_audit["manifest_sha256"] = sha256_file(manifest_path)
        if manifest is not None and manifest.get("schema") == "timeagent-retrieval-dev-pool-v1":
            path = directory / "targets.jsonl"
            digest = sha256_file(path)
            if digest != manifest.get("files", {}).get(path.name, {}).get("sha256"):
                raise ValueError("Development target pool does not match its manifest")
            rows = [row for _, row in _read_jsonl(path)]
            seen = set()
            for row in rows:
                identity = row.get("id")
                content = row.get("content")
                if not isinstance(identity, str) or not identity or identity in seen:
                    raise ValueError("Development pool target IDs must be nonempty and unique")
                if not isinstance(content, str) or body_hash(content) != row.get("body_sha256"):
                    raise ValueError("Development target body hash mismatch")
                url = canonical_url(row.get("canonical_url"))
                if not url:
                    raise ValueError("Development target requires a source URL")
                if {"gold", "golden_answers", "reward_model", "publication_date", "confirmed_publication_date"} & row.keys():
                    raise ValueError("Development pool must contain visible inputs without date labels")
                seen.add(identity); ids.add(identity); urls.add(url); bodies.add(row["body_sha256"])
            dataset_audit.update(schema=manifest["schema"], targets=len(rows), labels_read=False)
            dataset_audit["files"][path.name] = {"sha256": digest, "records": len(rows)}
            audit.append(dataset_audit)
            continue
        for split in ("train", "dev", "test"):
            path = directory / (split + ".json")
            digest = sha256_file(path)
            if manifest is not None and digest != manifest.get("files", {}).get(path.name, {}).get("sha256"):
                raise ValueError("Target dataset file does not match its manifest: " + str(path))
            rows = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                raise ValueError("Target dataset splits must be JSON lists")
            for row in rows:
                extra = row.get("extra_info", {})
                identity, digest_body = row.get("id"), extra.get("body_sha256")
                if not isinstance(identity, str) or not identity or not isinstance(digest_body, str) or not re.fullmatch(r"[a-f0-9]{64}", digest_body):
                    raise ValueError("Every excluded target requires id and extra_info.body_sha256")
                ids.add(identity); bodies.add(digest_body)
                url = canonical_url(extra.get("source_url"))
                if url:
                    urls.add(url)
            dataset_audit["files"][path.name] = {"sha256": digest, "records": len(rows)}
            dataset_audit["targets"] += len(rows)
        audit.append(dataset_audit)
    return ids, urls, bodies, audit


def _normalized_input(row):
    value = normalize_corpus_record(row)
    value.pop("rank", None)
    value["url"] = canonical_url(value["url"])
    digest = body_hash(value["content"])
    if "body_sha256" in value and value["body_sha256"] != digest:
        raise ValueError("Source body_sha256 does not match normalized source content")
    value["body_sha256"] = digest
    return value


def _prepare(corpora, dataset_dirs, report):
    excluded_ids, excluded_urls, excluded_bodies, target_audit = _target_exclusions(dataset_dirs)
    report["excluded_datasets"] = target_audit
    all_rows = []
    for source in corpora:
        path = Path(source).resolve()
        digest = sha256_file(path)
        source_audit = {"path": str(path), "sha256": digest, "records": 0}
        report["inputs"].append(source_audit)
        for line_number, original in _read_jsonl(path):
            origin = {"path": str(path), "line": line_number}
            report["input_records"] += 1
            source_audit["records"] += 1
            try:
                row = _normalized_input(original)
            except (ValueError, TypeError) as error:
                report["invalid_records"].append({**origin, "reason": str(error)})
                continue
            reasons = []
            if row["corpus_id"] in excluded_ids: reasons.append("target_id")
            if row["url"] and row["url"] in excluded_urls: reasons.append("target_url")
            if row["body_sha256"] in excluded_bodies: reasons.append("target_body")
            if reasons:
                report["excluded_records"].append({**origin, "corpus_id": row["corpus_id"], "reasons": reasons})
            else:
                all_rows.append((row, origin))
        if sha256_file(path) != digest:
            raise ValueError("Source corpus changed while it was being read: " + str(path))
    if report["invalid_records"]:
        raise ValueError("Invalid source records; no version was published")
    by_id, by_url, by_body = defaultdict(list), defaultdict(list), defaultdict(list)
    for row, origin in all_rows:
        by_id[row["corpus_id"]].append((row, origin))
        if row["url"]: by_url[row["url"]].append((row, origin))
        by_body[row["body_sha256"]].append((row, origin))
    for kind, groups in (("corpus_id", by_id), ("url", by_url), ("body_sha256", by_body)):
        for key, records in groups.items():
            bodies = {row["body_sha256"] for row, _ in records}
            dates = {row["publication_date"] for row, _ in records if row["publication_date"] is not None}
            urls = {row["url"] for row, _ in records}
            if len(dates) > 1 or kind in ("corpus_id", "url") and len(bodies) > 1 or kind == "corpus_id" and len(urls) > 1:
                report["conflicts"].append({"kind": kind, "key": key, "dates": sorted(dates),
                    "records": [{"corpus_id": row["corpus_id"], **origin} for row, origin in records]})
    if report["conflicts"]:
        raise ValueError("Conflicting source IDs, URL versions or publication dates; resolve recorded conflicts before building")
    retained = []
    for digest, records in sorted(by_body.items()):
        # Prefer a record carrying an observed date over a missing date. Never
        # combine conflicting dates; those groups have already failed above.
        ordered = sorted(records, key=lambda pair: (pair[0]["publication_date"] is None,
                         pair[0]["corpus_id"], json.dumps(pair[0], sort_keys=True, ensure_ascii=False)))
        selected, origin = ordered[0]
        retained.append(selected)
        for discarded, discarded_origin in ordered[1:]:
            report["deduplicated_records"].append({**discarded_origin, "corpus_id": discarded["corpus_id"],
                "retained_id": selected["corpus_id"], "retained_origin": origin,
                "body_sha256": digest, "reason": "same_normalized_body_and_nonconflicting_date"})
    if not retained:
        raise ValueError("No source evidence remains after exclusion and deduplication")
    return sorted(retained, key=lambda row: row["corpus_id"])


def _verify_directory(directory, expected_version):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != VERSION_SCHEMA or manifest.get("version") != expected_version:
        raise ValueError("Invalid corpus version manifest")
    if set(manifest.get("files", {})) != set(FILES):
        raise ValueError("Version manifest has an unexpected or missing file")
    for name, entry in manifest["files"].items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or sha256_file(path) != entry.get("sha256"):
            raise ValueError("Version file hash mismatch: " + name)
        if path.stat().st_size != entry.get("bytes"):
            raise ValueError("Version file size mismatch: " + name)
    rows = [_normalized_input(row) for _, row in _read_jsonl(directory / "corpus.jsonl")]
    if not rows or len(rows) != manifest.get("document_count"):
        raise ValueError("Version document count mismatch")
    for key in ("corpus_id", "body_sha256", "url"):
        values = [row[key] for row in rows if row[key]]
        if len(values) != len(set(values)):
            raise ValueError("Version corpus contains duplicate " + key)
    backend = LocalCorpusBackend(directory / "index.sqlite")
    if (backend.manifest.get("source_corpus_sha256") != manifest["files"]["corpus.jsonl"]["sha256"]
            or backend.manifest.get("document_count") != len(rows)):
        raise ValueError("Index and source corpus differ")
    if backend.manifest.get("source_corpus_path") != "corpus.jsonl":
        raise ValueError("Registry index must use a portable source corpus path")
    return {"status": "passed", "version": expected_version, "document_count": len(rows),
            "manifest_sha256": sha256_file(directory / "manifest.json"),
            "index": str(directory / "index.sqlite"), "backend_id": backend.backend_id,
            "source_corpus_sha256": manifest["files"]["corpus.jsonl"]["sha256"]}


def verify_version(registry, version=None):
    registry = Path(registry).resolve()
    current = None
    if version is None:
        current = json.loads((registry / "current.json").read_text(encoding="utf-8"))
        version = current.get("version")
        if current.get("schema") != CURRENT_SCHEMA or current.get("index") != "versions/" + _version(version) + "/index.sqlite":
            raise ValueError("Invalid current-version pointer")
    result = _verify_directory(_directory(registry, version), version)
    if current is not None and current.get("manifest_sha256") != result["manifest_sha256"]:
        raise ValueError("Active version manifest differs from current pointer")
    return result


def resolve_index(registry, version=None):
    return Path(verify_version(registry, version)["index"])


def _activate_unlocked(registry, version):
    checked = verify_version(registry, version)
    pointer = {"schema": CURRENT_SCHEMA, "version": version,
               "index": "versions/" + version + "/index.sqlite",
               "manifest_sha256": checked["manifest_sha256"], "activated_at_utc": _now()}
    temporary = registry / (".current-" + uuid.uuid4().hex + ".json")
    try:
        _write_json(temporary, pointer)
        os.replace(temporary, registry / "current.json")
    finally:
        if temporary.exists(): temporary.unlink()
    return {**checked, "activated": True}


def activate_version(registry, version):
    _version(version)
    with _writer_lock(registry) as registry:
        return _activate_unlocked(registry, version)


def list_versions(registry):
    registry = Path(registry).resolve()
    pointer = json.loads((registry / "current.json").read_text()) if (registry / "current.json").exists() else {}
    versions = []
    for directory in sorted((registry / "versions").glob("*")):
        if directory.name.startswith(".") or not directory.is_dir(): continue
        try:
            result = verify_version(registry, directory.name)
        except (ValueError, OSError, KeyError, TypeError) as error:
            result = {"status": "invalid", "version": directory.name, "error": str(error)}
        result["active"] = directory.name == pointer.get("version")
        versions.append(result)
    return {"registry": str(registry), "current_version": pointer.get("version"), "versions": versions}


def build_version(registry, version, corpora, exclude_dataset_dirs=(), activate=True):
    """Validate all inputs, build off-line, publish atomically, then activate.

    Invalid rows and source conflicts fail the entire build with a durable report.
    A failed build never changes current.json or exposes a partial version.
    """
    _version(version)
    corpora = list(corpora)
    if not corpora:
        raise ValueError("At least one source corpus JSONL is required")
    with _writer_lock(registry) as registry:
        destination = _directory(registry, version)
        if destination.exists():
            raise ValueError("Version already exists; choose a new version identifier")
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".build-" + version + "-", dir=destination.parent))
        report = {"schema": "timeagent-corpus-ingestion-v1", "version": version,
                  "created_at_utc": _now(), "status": "building", "inputs": [], "input_records": 0,
                  "excluded_datasets": [], "excluded_records": [], "invalid_records": [],
                  "conflicts": [], "deduplicated_records": [],
                  "limitations": "Exact normalized body/URL/ID checks only; no semantic or event clustering"}
        published = False
        try:
            rows = _prepare(corpora, exclude_dataset_dirs, report)
            with (stage / "corpus.jsonl").open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                stream.flush(); os.fsync(stream.fileno())
            index_manifest = build_index(stage / "corpus.jsonl", stage / "index.sqlite")
            index_manifest["source_corpus_path"] = "corpus.jsonl"
            index_manifest["source_corpus_path_base"] = "index_directory"
            _write_json(stage / "index.sqlite.manifest.json", index_manifest)
            report.update(status="passed", retained_records=len(rows),
                          excluded_count=len(report["excluded_records"]),
                          deduplicated_count=len(report["deduplicated_records"]))
            _write_json(stage / "ingestion_report.json", report)
            manifest = {"schema": VERSION_SCHEMA, "version": version, "document_count": len(rows),
                        "created_at_utc": _now(), "immutable": True,
                        "files": {name: {"sha256": sha256_file(stage / name), "bytes": (stage / name).stat().st_size}
                                  for name in FILES}}
            _write_json(stage / "manifest.json", manifest)
            _verify_directory(stage, version)
            os.rename(stage, destination)
            published = True
            result = _activate_unlocked(registry, version) if activate else verify_version(registry, version)
            result["activated"] = activate
            result["ingestion_report"] = str(destination / "ingestion_report.json")
            return result
        except Exception as error:
            report.update(status="failed", error_type=type(error).__name__, error=str(error), published=published)
            failed = registry / "failed"
            failed.mkdir(exist_ok=True)
            failure_path = failed / (version + "-" + uuid.uuid4().hex + ".json")
            _write_json(failure_path, report)
            raise RegistryBuildError(str(error), failure_path) from error
        finally:
            if stage.exists(): shutil.rmtree(stage)
