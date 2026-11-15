"""Strict, relocatable configuration for the retrieval service and its launcher."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA = "timeagent-retrieval-service-v1"
SERVER_DEFAULTS = {
    "bind": "127.0.0.1", "port": 8000, "workers": 8,
    "max_request_bytes": 1048576, "max_queries": 512,
    "max_query_chars": 4096, "max_concurrent_requests": 16,
    "request_timeout": 15.0, "token_env": None,
}
RETRIEVAL_DEFAULTS = {"topk": 3, "view": "full", "seed": 20261103,
                      "cache_size": 1024, "backend_timeout": 10.0}


def object_fields(value, allowed, name):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f"Invalid {name} object or unknown configuration fields")
    return value


def resolve_path(value, base, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty path")
    path = Path(value).expanduser()
    return str((base / path).resolve() if not path.is_absolute() else path.resolve())


def load_config(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    data = json.loads(raw)
    object_fields(data, {"schema", "backend", "server", "retrieval", "runs_dir"}, "root")
    if data.get("schema") != SCHEMA:
        raise ValueError("Unsupported service configuration schema")
    backend = dict(object_fields(data.get("backend"),
                   {"kind", "registry", "version", "index", "snapshot", "host"}, "backend"))
    kind = backend.get("kind")
    allowed = {"local": {"kind", "registry", "version", "index"},
               "replay": {"kind", "snapshot"}, "aiapi": {"kind", "host"}}
    if kind not in allowed or set(backend) - allowed[kind]:
        raise ValueError("Invalid backend kind or incompatible backend options")
    if kind == "local":
        if ("registry" in backend) == ("index" in backend):
            raise ValueError("Local backend requires exactly one of registry or index")
        if "version" in backend and "registry" not in backend:
            raise ValueError("version requires a registry")
        if backend.get("version") is not None and (not isinstance(backend["version"], str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", backend["version"])):
            raise ValueError("Invalid corpus version")
    if kind == "replay" and "snapshot" not in backend:
        raise ValueError("Replay requires a snapshot path")
    for key in ("registry", "index", "snapshot"):
        if key in backend:
            backend[key] = resolve_path(backend[key], path.parent, key)
    if "host" in backend:
        from .web_server import backend_host
        backend["host"] = backend_host(backend["host"])
    server = {**SERVER_DEFAULTS, **object_fields(data.get("server", {}), SERVER_DEFAULTS, "server")}
    retrieval = {**RETRIEVAL_DEFAULTS,
                 **object_fields(data.get("retrieval", {}), RETRIEVAL_DEFAULTS, "retrieval")}
    if not isinstance(server["bind"], str) or not server["bind"].strip() or "/" in server["bind"]:
        raise ValueError("bind must be a hostname or IP address")
    for key in ("port", "workers", "max_request_bytes", "max_queries", "max_query_chars", "max_concurrent_requests"):
        if type(server[key]) is not int or server[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if server["port"] > 65535:
        raise ValueError("port must be <=65535")
    for key in ("topk", "cache_size"):
        if type(retrieval[key]) is not int or retrieval[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if retrieval["topk"] > 100 or type(retrieval["seed"]) is not int or retrieval["seed"] < 0:
        raise ValueError("topk must be <=100 and seed must be nonnegative")
    if retrieval["view"] not in ("full", "text", "dates", "shuffled-dates"):
        raise ValueError("Invalid evidence view")
    for value in (server["request_timeout"], retrieval["backend_timeout"]):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Timeouts must be finite positive numbers")
    token_env = server["token_env"]
    if token_env is not None and (not isinstance(token_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token_env)):
        raise ValueError("token_env must be an environment variable name, never a token value")
    return {"schema": SCHEMA, "config_path": str(path), "config_sha256": hashlib.sha256(raw).hexdigest(),
            "backend": backend, "server": server, "retrieval": retrieval,
            "runs_dir": resolve_path(data.get("runs_dir", "runs"), path.parent, "runs_dir")}


def service_url(config):
    bind = config["server"]["bind"]
    host = "127.0.0.1" if bind == "0.0.0.0" else "::1" if bind == "::" else bind
    if ":" in host:
        host = "[" + host + "]"
    return f"http://{host}:{config['server']['port']}"


def resolve_backend(config):
    backend = dict(config["backend"])
    if backend["kind"] == "local" and "registry" in backend:
        from .corpus_registry import resolve_index
        backend["index"] = str(resolve_index(backend["registry"], backend.get("version")))
    return backend


def service_args(config, backend, snapshot, instance_id):
    args = ["--snapshot", str(snapshot), "--instance-id", instance_id]
    if backend["kind"] == "replay":
        args += ["--replay-snapshot", backend["snapshot"]]
    else:
        args += ["--backend", backend["kind"]]
        if backend["kind"] == "local":
            args += ["--index", backend["index"]]
        elif backend.get("host"):
            args += ["--host", backend["host"]]
    for key, value in config["server"].items():
        if value is not None:
            args += ["--" + key.replace("_", "-"), str(value)]
    for key, value in config["retrieval"].items():
        flag = "timeout" if key == "backend_timeout" else key.replace("_", "-")
        args += ["--" + flag, str(value)]
    return args
