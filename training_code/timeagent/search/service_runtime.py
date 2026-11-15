"""Bounded HTTP runtime and process-local operational statistics."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from http.server import HTTPServer
import json
import math
import socket
from socketserver import ThreadingMixIn
import threading
import time
import uuid


@dataclass(frozen=True)
class ServiceLimits:
    max_request_bytes: int = 1_048_576
    max_queries: int = 512
    max_query_chars: int = 4096
    max_concurrent_requests: int = 16
    request_timeout: float = 15.0

    def __post_init__(self):
        for name in ('max_request_bytes', 'max_queries', 'max_query_chars', 'max_concurrent_requests'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + ' must be a positive integer')
        if isinstance(self.request_timeout, bool) or not isinstance(self.request_timeout, (int, float)) or not math.isfinite(self.request_timeout) or self.request_timeout <= 0:
            raise ValueError('request_timeout must be finite and positive')


class ServiceStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.counters = Counter()
        self.statuses = Counter()
        self.routes = Counter()
        self.request_seconds_total = 0.0
        self.request_seconds_max = 0.0
        self.query_seconds_total = 0.0

    def add(self, name, amount=1):
        with self.lock:
            self.counters[name] += amount

    def request_finished(self, route, status, elapsed):
        with self.lock:
            self.counters['requests_completed'] += 1
            self.counters['request_errors'] += int(status >= 400)
            self.statuses[str(status)] += 1
            self.routes[route if route in ('/retrieve', '/healthz', '/readyz', '/stats') else 'other'] += 1
            self.request_seconds_total += elapsed
            self.request_seconds_max = max(self.request_seconds_max, elapsed)

    def query_finished(self, elapsed, failed):
        with self.lock:
            self.counters['queries_completed'] += 1
            self.counters['query_errors'] += int(failed)
            self.query_seconds_total += elapsed

    def snapshot(self):
        with self.lock:
            return {'uptime_seconds': time.monotonic() - self.started,
                    'counters': dict(self.counters), 'http_status_counts': dict(self.statuses),
                    'route_counts': dict(self.routes),
                    'request_seconds_total': self.request_seconds_total,
                    'request_seconds_max': self.request_seconds_max,
                    'query_seconds_total': self.query_seconds_total,
                    'latency_definition': 'server wall time; sums include concurrently running requests/queries'}


class BoundedHTTPServer(ThreadingMixIn, HTTPServer):
    """Reject excess connections before creating threads; socket reads time out."""
    daemon_threads = False
    block_on_close = True
    allow_reuse_address = True
    request_queue_size = 64

    def __init__(self, address, handler, limits, stats):
        self.limits, self.stats = limits, stats
        self.slots = threading.BoundedSemaphore(limits.max_concurrent_requests)
        if ':' in address[0]:
            self.address_family = socket.AF_INET6
        super().__init__(address, handler)

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(self.limits.request_timeout)
        return connection, address

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            payload = json.dumps({'error': 'retrieval service is busy', 'request_id': uuid.uuid4().hex}).encode()
            try:
                request.sendall(b'HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nConnection: close\r\nRetry-After: 1\r\nContent-Length: ' + str(len(payload)).encode() + b'\r\n\r\n' + payload)
            except (OSError, socket.timeout):
                pass
            finally:
                self.stats.add('overload_rejections')
                self.stats.request_finished('other', 503, 0)
                self.shutdown_request(request)
            return
        self.stats.add('active_connections')
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            self.stats.add('active_connections', -1)
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()
            self.stats.add('active_connections', -1)

    def handle_error(self, request, client_address):
        # Avoid writing request/backend exception payloads to the console.
        self.stats.add('uncaught_connection_errors')
