

"""Retrieval HTTP server with frozen evidence views.

Keys are environment-only. Importing does not start a server or use the network.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from contextlib import closing
import hmac
import signal
import sqlite3
import time
import uuid
import datetime as dt
import hashlib
from http.server import BaseHTTPRequestHandler
import json
import math
import os
from pathlib import Path
import re
import socket
import threading
import urllib.parse
import urllib.request

from timeagent.evidence import VIEWS, canonical_record, evidence_view, pool_fingerprint
from timeagent.search.local_backend import BackendError, LocalCorpusBackend
from timeagent.search.service_runtime import BoundedHTTPServer, ServiceLimits, ServiceStats

SNAPSHOT_SCHEMA = "timeagent-search-snapshot-v1"


def api_auth_sign(params):
    """Preserve the existing AIAPI MD5/selected-position protocol."""
    digest = hashlib.md5("".join(params).encode("utf-8")).hexdigest()
    return "".join(digest[i] for i in (7, 3, 17, 13, 1, 21))


def sanitize_search_query(query):
    if not isinstance(query, str):
        raise ValueError("Each query must be a string")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", query)).strip()


def backend_host(host):
    if not isinstance(host, str) or not host:
        raise ValueError("Provide --host or AIAPI_HOST; no BNS fallback is used")
    parsed = urllib.parse.urlsplit(host)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Backend host must be an HTTP(S) URL without credentials/query/fragment")
    return host.rstrip("/")


class AIAPIBackend:
    def __init__(self, host, ak, sk, timeout=10, client_ip=None, opener=None):
        self.backend_id = backend_host(host)
        if not ak or not sk:
            raise ValueError("Set AIAPI_AK and AIAPI_SK in the environment")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self._ak, self._sk = ak, sk
        self.timeout = timeout
        self.opener = opener or urllib.request.urlopen
        if client_ip is None:
            try:
                client_ip = socket.gethostbyname(socket.gethostname())
            except OSError:
                client_ip = "127.0.0.1"
        self.client_ip = client_ip

    def search(self, query, topk, max_date=None):
        url = self.backend_id + "/search/api/v2?" + urllib.parse.urlencode({"word": query, "clientip": self.client_ip})
        request = urllib.request.Request(url, data=json.dumps({"size": {"main": topk, "video": 0}}).encode(),
                                         headers={"Api-Key": self._ak, "Content-Type": "application/json",
                                                  "Api-Auth": api_auth_sign([self._ak, query, self.client_ip, self._sk])})
        try:
            with self.opener(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as error:
            # Do not expose backend exception strings or secret-bearing headers.
            raise BackendError(f"AIAPI request failed ({type(error).__name__})") from None
        if not isinstance(payload, dict):
            raise BackendError("AIAPI returned a non-object response")
        if payload.get("error_code", 0) != 0:
            raise BackendError("AIAPI returned an error status")
        results = payload.get("results")
        if not isinstance(results, dict) or not isinstance(results.get("result"), list):
            raise BackendError("AIAPI response is missing results.result")
        try:
            return [canonical_record(item, i + 1) for i, item in enumerate(results["result"][:topk])]
        except ValueError:
            raise BackendError("AIAPI returned a malformed result record") from None


class SnapshotReplayBackend:
    """Exact-query replay; unknown queries fail without an online fallback."""
    def __init__(self, path):
        path = Path(path)
        self.backend_id = "snapshot-sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        self.records = {}
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != SNAPSHOT_SCHEMA:
                raise ValueError(f"Unknown snapshot schema at line {line_no}")
            if row.get("status") != "ok":
                continue
            key = (row["query_sanitized"], row["requested_topk"])
            pool = row["raw_results"]
            if row.get("pool_sha256") != pool_fingerprint(pool):
                raise ValueError(f"Snapshot pool hash mismatch at line {line_no}")
            if key in self.records and pool_fingerprint(self.records[key]) != row["pool_sha256"]:
                raise ValueError("Snapshot has conflicting pools for one query/topk")
            self.records[key] = pool
        if not self.records:
            raise ValueError("Snapshot contains no successful evidence pools")

    def search(self, query, topk, max_date=None):
        key = (query, topk)
        if key not in self.records:
            raise BackendError("Query/topk is absent from the frozen snapshot")
        return self.records[key]


class SnapshotWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def append(self, row):
        record = {"schema": SNAPSHOT_SCHEMA,
                  "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), **row}
        try:
            encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
            with self.lock, self.path.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
                stream.flush()
        except (OSError, ValueError, TypeError):
            raise BackendError("Could not persist the retrieval snapshot") from None


class CacheCapacityError(BackendError):
    """Mutable backend cannot accept another query without losing frozen pools."""


class OnlineSearchEngine:
    def __init__(self, backend, snapshot, topk=3, view="full", seed=20261103, workers=16,
                 cache_size=1024, limits=None, stats=None, instance_id=None):
        if type(topk) is not int or not 1 <= topk <= 100:
            raise ValueError("topk must be an integer in [1,100]")
        if view not in VIEWS or type(seed) is not int or type(workers) is not int or workers < 1:
            raise ValueError("Invalid view, seed, or workers")
        if type(cache_size) is not int or cache_size < 1:
            raise ValueError("cache_size must be a positive integer")
        if instance_id is not None and (not isinstance(instance_id, str) or not instance_id.strip() or len(instance_id) > 128):
            raise ValueError("instance_id must be nonempty text of at most 128 characters")
        self.instance_id = instance_id or uuid.uuid4().hex
        self.backend, self.writer = backend, SnapshotWriter(snapshot)
        self.topk, self.view, self.seed, self.workers = topk, view, seed, workers
        self.limits, self.stats = limits or ServiceLimits(), stats or ServiceStats()
        self.cache, self.cache_size = OrderedDict(), cache_size
        self.cache_lock = threading.Lock()
        # Fixed stripes bound synchronization state even after millions of keys.
        self.key_locks = tuple(threading.Lock() for _ in range(256))
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="retrieval-query")
        self.backend_slots = threading.BoundedSemaphore(workers)
        self.inflight = set()
        self.freeze_mutable_backend = isinstance(backend, AIAPIBackend)
        self.closed = False

    def close(self):
        with self.cache_lock:
            self.closed = True
        self.executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def readiness(self):
        if self.closed:
            raise BackendError("Retrieval engine is shutting down")
        info = {"status": "ready", "instance_id": self.instance_id, "backend_id": self.backend.backend_id,
                "view": self.view, "default_topk": self.topk}
        if isinstance(self.backend, LocalCorpusBackend):
            try:
                if self.backend._file_signature() != self.backend._stat:
                    raise BackendError("Frozen local index changed after startup")
                with closing(self.backend._connect()) as db:
                    db.execute("SELECT rowid FROM corpus_fts LIMIT 1").fetchone()
                info["document_count"] = self.backend.manifest["document_count"]
                info["index_sha256"] = self.backend.manifest["index_sha256"]
            except (OSError, sqlite3.Error):
                raise BackendError("Local index is not readable") from None
        elif isinstance(self.backend, SnapshotReplayBackend):
            info["frozen_query_count"] = len(self.backend.records)
        else:
            info["readiness_scope"] = "configured; no external API probe"
        return info

    def stats_snapshot(self):
        value = self.stats.snapshot()
        with self.cache_lock:
            value.update({"cache_entries": len(self.cache), "cache_capacity": self.cache_size,
                          "backend_queries_inflight": len(self.inflight),
                          "cache_policy": "freeze-or-reject" if self.freeze_mutable_backend else "LRU",
                          "workers": self.workers})
        return value

    def _query(self, query, topk):
        if type(topk) is not int or not 1 <= topk <= 100:
            raise ValueError("topk must be an integer in [1,100]")
        if not isinstance(query, str) or not query or len(query) > self.limits.max_query_chars:
            raise ValueError("Query must be nonempty text within the configured character limit")
        sanitized = sanitize_search_query(query)
        if not sanitized:
            raise ValueError("Query must contain searchable text")
        return sanitized

    def search(self, query, topk=None, *, max_date=None, request_id=None):
        topk = self.topk if topk is None else topk
        if max_date is not None:
            if not isinstance(max_date, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", max_date):
                raise ValueError("max_date must be a YYYY-MM-DD string")
            try:
                dt.date.fromisoformat(max_date)
            except ValueError:
                raise ValueError("max_date must be a real calendar date") from None
        sanitized = self._query(query, topk)
        if request_id is not None and (not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", request_id)):
            raise ValueError("Invalid request_id")
        key = (sanitized, topk, max_date)
        metadata = {"query": query, "query_sanitized": sanitized, "requested_topk": topk,
                    "backend": self.backend.backend_id, "view": self.view, "seed": self.seed,
                    "max_date": max_date}
        if request_id is not None:
            metadata["request_id"] = request_id
        started, failed = time.monotonic(), True
        try:
            # Check even on a cache hit: changed local indexes must fail closed.
            self.readiness()
            with self.key_locks[hash(key) % len(self.key_locks)]:
                with self.cache_lock:
                    hit = key in self.cache
                    if hit:
                        pool = self.cache[key]
                        self.cache.move_to_end(key)
                    else:
                        if self.freeze_mutable_backend and len(self.cache) + len(self.inflight) >= self.cache_size:
                            raise CacheCapacityError("Frozen online query cache is full; start a new versioned run")
                        self.inflight.add(key)
                self.stats.add("cache_hits" if hit else "cache_misses")
                if not hit:
                    try:
                        with self.backend_slots:
                            self.stats.add("backend_queries")
                            pool = self.backend.search(sanitized, topk, max_date=max_date)
                        with self.cache_lock:
                            self.cache[key] = pool
                            if len(self.cache) > self.cache_size:
                                self.cache.popitem(last=False)
                                self.stats.add("cache_evictions")
                    finally:
                        with self.cache_lock:
                            self.inflight.discard(key)
                rendered = evidence_view(pool, self.view, self.seed, sanitized)
            self.writer.append({**metadata, "status": "ok", "cache_hit": hit,
                                "pool_sha256": pool_fingerprint(pool), "raw_results": pool,
                                "rendered_results": rendered})
            failed = False
            return rendered
        except BackendError as error:
            self.writer.append({**metadata, "status": "error", "error": type(error).__name__})
            raise
        except ValueError:
            # Query-specific backend limits retain HTTP 400 semantics.
            raise
        except Exception as error:
            self.writer.append({**metadata, "status": "error", "error": type(error).__name__})
            raise BackendError("Unexpected backend failure") from None
        finally:
            self.stats.query_finished(time.monotonic() - started, failed)

    def batch_search(self, queries, topk=None, *, max_date=None, request_id=None):
        if not isinstance(queries, list) or not queries or len(queries) > self.limits.max_queries:
            raise ValueError("queries exceeds configured batch limit or is empty")
        effective_topk = self.topk if topk is None else topk
        for query in queries:
            self._query(query, effective_topk)
        if request_id is not None and (not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", request_id)):
            raise ValueError("Invalid request_id")
        if self.closed:
            raise BackendError("Retrieval engine is shutting down")
        # One process-wide executor, instead of a new worker pool per HTTP call.
        futures = [self.executor.submit(self.search, query, effective_topk, max_date=max_date, request_id=request_id) for query in queries]
        # Drain accepted work before returning a batch error; no hidden work lingers.
        values, errors = [], []
        for future in futures:
            try:
                values.append(future.result())
            except Exception as error:
                values.append(None)
                errors.append(error)
        if errors:
            raise errors[0]
        return values


def _strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("Nonfinite JSON constant")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def make_handler(engine, token=None):
    class RetrievalHandler(BaseHTTPRequestHandler):
        # Close every connection so one keep-alive cannot monopolize a worker.
        protocol_version = "HTTP/1.0"
        server_version = "TIME-Agent-Retrieval/1"
        sys_version = ""

        def log_message(self, format, *args):
            return

        def _start(self):
            self._request_started = time.monotonic()
            self._request_id = uuid.uuid4().hex

        def _respond(self, status, payload):
            if not hasattr(self, "_request_started"):
                self._start()
            value = {**payload, "request_id": self._request_id}
            try:
                self.send_json(status, value)
            except (OSError, socket.timeout):
                engine.stats.add("client_disconnects")
            finally:
                engine.stats.request_finished(self.path, status, time.monotonic() - self._request_started)

        def send_json(self, status, payload):
            encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            if "request_id" in payload:
                self.send_header("X-Request-ID", payload["request_id"])
            self.end_headers()
            self.wfile.write(encoded)

        def send_error(self, code, message=None, explain=None):
            self._respond(code, {"error": "invalid HTTP request"})

        def _authorized(self):
            if token is None:
                return True
            supplied = self.headers.get("Authorization", "")
            if not isinstance(supplied, str) or not hmac.compare_digest(supplied.encode(), ("Bearer " + token).encode()):
                self._respond(401, {"error": "authorization required"})
                return False
            return True

        def do_GET(self):
            self._start()
            if self.path == "/healthz":
                self._respond(200, {"status": "alive", "instance_id": engine.instance_id})
                return
            if not self._authorized():
                return
            if self.path == "/stats":
                self._respond(200, engine.stats_snapshot())
            elif self.path == "/readyz":
                try:
                    self._respond(200, engine.readiness())
                except Exception:
                    self._respond(503, {"status": "not_ready", "error": "backend unavailable"})
            else:
                self._respond(404, {"error": "unknown endpoint"})

        def do_POST(self):
            self._start()
            if self.path != "/retrieve":
                self._respond(404, {"error": "unknown endpoint"})
                return
            if not self._authorized():
                return
            try:
                # No chunked framing, duplicate lengths, or ambiguous lengths.
                lengths = self.headers.get_all("Content-Length", []) if hasattr(self.headers, "get_all") else [self.headers.get("Content-Length", "")]
                if self.headers.get("Transfer-Encoding") is not None or len(lengths) != 1 or not re.fullmatch(r"[0-9]+", lengths[0]):
                    raise ValueError("Invalid request framing")
                length = int(lengths[0])
                if length > engine.limits.max_request_bytes:
                    self._respond(413, {"error": "request body exceeds configured limit"})
                    return
                if length < 1:
                    raise ValueError("Empty request")
                media_type = self.headers.get("Content-Type", "application/json").split(";", 1)[0].strip().lower()
                if media_type != "application/json":
                    self._respond(415, {"error": "expected application/json"})
                    return
                # A body deadline also bounds clients that drip bytes often
                # enough to avoid a per-read socket timeout.
                deadline = time.monotonic() + engine.limits.request_timeout
                chunks, remaining = [], length
                while remaining:
                    timeout = deadline - time.monotonic()
                    if timeout <= 0:
                        raise TimeoutError()
                    if hasattr(self, "connection"):
                        self.connection.settimeout(timeout)
                    read = getattr(self.rfile, "read1", self.rfile.read)
                    chunk = read(min(remaining, 65536))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                encoded = b"".join(chunks)
                if len(encoded) != length:
                    raise ValueError("Incomplete request body")
                if hasattr(self, "connection"):
                    self.connection.settimeout(engine.limits.request_timeout)
                payload = _strict_json(encoded.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object")
                results = engine.batch_search(payload.get("queries"), payload.get("topk"),
                                              max_date=payload.get("max_date"), request_id=self._request_id)
            except (TimeoutError, socket.timeout):
                self._respond(408, {"error": "request read timed out"})
                return
            except (ValueError, TypeError, UnicodeDecodeError):
                self._respond(400, {"error": "invalid retrieval request"})
                return
            except CacheCapacityError:
                self._respond(503, {"error": "frozen online query capacity reached"})
                return
            except BackendError:
                self._respond(502, {"error": "retrieval failed; consult the snapshot error record"})
                return
            except Exception:
                engine.stats.add("unexpected_request_errors")
                self._respond(500, {"error": "internal retrieval error"})
                return
            self._respond(200, {"result": results})
    return RetrievalHandler


def main(argv=None, on_ready=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("local", "aiapi"),
                        help="Required unless replaying a snapshot; never silently defaults to an online backend")
    parser.add_argument("--index", type=Path, help="Frozen local SQLite index for --backend local")
    parser.add_argument("--host", default=os.environ.get("AIAPI_HOST"), help="AIAPI backend base URL")
    parser.add_argument("--bind", default="127.0.0.1", help="Local listen address")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--view", choices=VIEWS, default="full")
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--cache-size", type=int, default=1024)
    parser.add_argument("--max-request-bytes", type=int, default=1_048_576)
    parser.add_argument("--max-queries", type=int, default=512)
    parser.add_argument("--max-query-chars", type=int, default=4096)
    parser.add_argument("--max-concurrent-requests", type=int, default=16)
    parser.add_argument("--request-timeout", type=float, default=15.0)
    parser.add_argument("--token-env", help="Optional environment variable holding the service Bearer token")
    parser.add_argument("--instance-id", help="Optional instance identity exposed by health/readiness")
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--snapshot", required=True, help="Append-only JSONL log for this run")
    parser.add_argument("--replay-snapshot", help="Frozen pool only; no network or keys needed")
    args = parser.parse_args(argv)
    engine = None
    try:
        limits = ServiceLimits(args.max_request_bytes, args.max_queries, args.max_query_chars,
                               args.max_concurrent_requests, args.request_timeout)
        token = os.environ.get(args.token_env) if args.token_env else None
        if args.token_env and (not token or any(ord(c) < 33 or ord(c) > 126 for c in token)):
            raise ValueError("Configured token environment variable is missing or invalid")
        if not 0 <= args.port <= 65535:
            raise ValueError("port must be in [0,65535]")
        if args.replay_snapshot:
            if args.backend or args.index:
                raise ValueError("--replay-snapshot cannot be combined with --backend or --index")
            if Path(args.replay_snapshot).resolve() == Path(args.snapshot).resolve():
                raise ValueError("Replay input and output snapshot paths must differ")
            backend = SnapshotReplayBackend(args.replay_snapshot)
        elif args.backend == "local":
            if not args.index:
                raise ValueError("--backend local requires --index")
            backend = LocalCorpusBackend(args.index)
        elif args.backend == "aiapi":
            if args.index:
                raise ValueError("--index is only valid with --backend local")
            backend = AIAPIBackend(args.host, os.environ.get("AIAPI_AK"), os.environ.get("AIAPI_SK"),
                                   args.timeout, os.environ.get("AIAPI_CLIENT_IP"))
        else:
            raise ValueError("Choose --backend local/aiapi, or --replay-snapshot")
        engine = OnlineSearchEngine(backend, args.snapshot, args.topk, args.view, args.seed, args.workers,
                                    args.cache_size, limits, instance_id=args.instance_id)
        server = BoundedHTTPServer((args.bind, args.port), make_handler(engine, token), limits, engine.stats)
    except (ValueError, OSError) as error:
        if engine is not None:
            engine.close()
        parser.exit(2, f"serve_search: {error}\n")
    print(f"Retrieval service listening on {args.bind}:{server.server_port}; view={args.view}; instance_id={engine.instance_id}; snapshot enabled", flush=True)
    previous_signal = None
    stopping = threading.Event()
    def stop(signum, frame):
        if not stopping.is_set():
            stopping.set()
            threading.Thread(target=server.shutdown, name="retrieval-shutdown", daemon=True).start()
    if threading.current_thread() is threading.main_thread():
        previous_signal = signal.signal(signal.SIGTERM, stop)
    try:
        if on_ready is not None:
            on_ready({**engine.readiness(), "bind": args.bind, "port": server.server_port})
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.close()
        if previous_signal is not None:
            signal.signal(signal.SIGTERM, previous_signal)


if __name__ == "__main__":
    main()