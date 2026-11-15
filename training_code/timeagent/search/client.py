"""Dependency-free retrieval client with explicit failures and no retries.

The service may have executed a query before an HTTP timeout, so automatically
retrying would make the training executor's per-query accounting ambiguous.
"""
from __future__ import annotations

import json
import http.client
import math
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request


class RetrievalClientError(RuntimeError):
    """Safe service/transport error; raw response bodies are never included."""
    kind = "retrieval_error"

    def as_dict(self):
        result = {"error": self.kind, "message": str(self)}
        if getattr(self, "status", None) is not None:
            result["http_status"] = self.status
        if getattr(self, "request_id", None):
            result["request_id"] = self.request_id
        return result


class RetrievalTimeout(RetrievalClientError):
    kind = "timeout"


class RetrievalConnectionError(RetrievalClientError):
    kind = "connection_error"


class RetrievalHTTPError(RetrievalClientError):
    kind = "http_error"

    def __init__(self, status, request_id=None):
        self.status = status
        self.request_id = request_id
        super().__init__(f"Retrieval service returned HTTP {status}")


class RetrievalProtocolError(RetrievalClientError):
    kind = "invalid_response"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the optional Authorization header to another endpoint.
        return None


def normalize_service_url(value):
    if not isinstance(value, str) or not value or any(ch.isspace() or ord(ch) < 32 for ch in value):
        raise ValueError("Retrieval URL must be an HTTP(S) URL without whitespace")
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment or (port is not None and port <= 0)):
            raise ValueError
    except ValueError:
        raise ValueError("Retrieval URL must be HTTP(S), with no credentials, query or fragment") from None
    path = parsed.path.rstrip("/")
    if path.endswith("/retrieve"):
        path = path[:-len("/retrieve")]
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def validate_client_options(timeout=30, token_env=None, max_response_bytes=20_000_000):
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Retrieval timeout must be finite and positive")
    if type(max_response_bytes) is not int or max_response_bytes <= 0:
        raise ValueError("max_response_bytes must be a positive integer")
    if token_env is not None and (not isinstance(token_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token_env)):
        raise ValueError("token_env must name an environment variable")


def _safe_request_id(value):
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value) else None


class RetrievalClient:
    def __init__(self, url="http://127.0.0.1:8000", timeout=30, topk=3,
                 token_env=None, max_response_bytes=20_000_000, opener=None):
        self.base_url = normalize_service_url(url)
        validate_client_options(timeout, token_env, max_response_bytes)
        self._validate_topk(topk)
        self.timeout = timeout
        self.topk = topk
        self.max_response_bytes = max_response_bytes
        self._token = None
        if token_env is not None:
            token = os.environ.get(token_env)
            if not token or any(ord(ch) < 33 or ord(ch) > 126 for ch in token):
                raise ValueError("Retrieval token environment variable is missing, empty or malformed")
            self._token = token
        self._open = opener or urllib.request.build_opener(NoRedirect()).open

    @staticmethod
    def _validate_topk(topk):
        if type(topk) is not int or not 1 <= topk <= 100:
            raise ValueError("topk must be an integer in [1,100]")

    def _request(self, path, payload=None, authenticated=True):
        headers = {"Accept": "application/json"}
        if authenticated and self._token:
            headers["Authorization"] = "Bearer " + self._token
        raw = None
        if payload is not None:
            raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.base_url + path, data=raw, headers=headers)
        try:
            with self._open(request, timeout=self.timeout) as response:
                status = getattr(response, "status", 200)
                if status != 200:
                    raise RetrievalHTTPError(status)
                content = response.read(self.max_response_bytes + 1)
        except urllib.error.HTTPError as error:
            # Header IDs are useful for server log lookup; never expose reason/body.
            request_id = _safe_request_id(error.headers.get("X-Request-ID")) if error.headers else None
            if request_id and self._token and self._token in request_id:
                request_id = None
            status = error.code
            error.close()
            raise RetrievalHTTPError(status, request_id) from None
        except (TimeoutError, socket.timeout):
            raise RetrievalTimeout("Retrieval service request timed out; no automatic retry was made") from None
        except urllib.error.URLError as error:
            if isinstance(error.reason, (TimeoutError, socket.timeout)):
                raise RetrievalTimeout("Retrieval service request timed out; no automatic retry was made") from None
            raise RetrievalConnectionError("Could not connect to the retrieval service") from None
        except OSError:
            raise RetrievalConnectionError("Could not read from the retrieval service") from None
        except http.client.HTTPException:
            raise RetrievalProtocolError("Retrieval HTTP response was incomplete or malformed") from None
        if len(content) > self.max_response_bytes:
            raise RetrievalProtocolError("Retrieval response exceeded max_response_bytes")
        try:
            result = json.loads(content.decode("utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        except (UnicodeDecodeError, ValueError, RecursionError):
            raise RetrievalProtocolError("Retrieval service returned invalid JSON") from None
        if not isinstance(result, dict):
            raise RetrievalProtocolError("Retrieval response must be a JSON object")
        return result

    def retrieve(self, queries, topk=None):
        topk = self.topk if topk is None else topk
        self._validate_topk(topk)
        if not isinstance(queries, list) or not 1 <= len(queries) <= 4096:
            raise ValueError("queries must be a list of 1..4096 strings")
        if any(not isinstance(q, str) or not q.strip() or len(q) > 65536 for q in queries):
            raise ValueError("Each query must be nonempty text of at most 65536 characters")
        result = self._request("/retrieve", {"queries": queries, "topk": topk, "return_scores": True})
        rows = result.get("result")
        if not isinstance(rows, list) or len(rows) != len(queries):
            raise RetrievalProtocolError("Retrieval result count differs from the dispatched query count")
        for group in rows:
            if not isinstance(group, list) or len(group) > topk:
                raise RetrievalProtocolError("Retrieval response contains an invalid result group")
            for row in group:
                if (not isinstance(row, dict) or not isinstance(row.get("document"), dict)
                        or not isinstance(row["document"].get("contents"), str)):
                    raise RetrievalProtocolError("Retrieved document lacks text contents")
        return result

    def health(self):
        return self._request("/healthz", authenticated=False)

    def ready(self):
        return self._request("/readyz")

    def stats(self):
        return self._request("/stats")
