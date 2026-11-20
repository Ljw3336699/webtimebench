#!/usr/bin/env bash
# End-to-end example: build an index, serve it, record a snapshot, then replay the
# snapshot offline and show the evidence returned is byte-identical.
#
# Needs only the Python standard library. Runs in a few minutes on the shipped
# 50-event WCEP pilot. Usage: bash examples/replay_equivalence.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${TMPDIR:-/tmp}/webtimebench_replay_demo"
PY="${PYTHON:-python3}"
rm -rf "$WORK"; mkdir -p "$WORK"
cd "$ROOT"

echo "[1/5] build a frozen index from the shipped pilot corpus"
"$PY" scripts/build_local_index.py \
  --corpus events/wcep_pilot/corpus.jsonl \
  --output "$WORK/pilot.sqlite"
"$PY" - "$WORK/pilot.sqlite.manifest.json" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
print("      documents:", m["document_count"], "index sha256:", m["index_sha256"][:16])
PYEOF

echo "[2/5] serve the index and record every request to a snapshot"
PYTHONPATH=src "$PY" scripts/serve_search.py --backend local --index "$WORK/pilot.sqlite" \
  --port 8077 --view full --snapshot "$WORK/live.jsonl" > "$WORK/serve_live.log" 2>&1 &
LIVE=$!
trap 'kill $LIVE 2>/dev/null || true' EXIT
for _ in $(seq 30); do grep -q listening "$WORK/serve_live.log" && break; sleep 1; done

QUERIES='["flooding in Louisiana","president signs executive order","earthquake relief effort"]'
"$PY" - "$QUERIES" "$WORK/live_results.json" <<'PYEOF'
import json, sys, urllib.request
queries = json.loads(sys.argv[1])
body = json.dumps({"queries": queries, "topk": 3}).encode()
req = urllib.request.Request("http://127.0.0.1:8077/retrieve", data=body,
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=60) as response:
    payload = json.load(response)
json.dump(payload, open(sys.argv[2], "w"), sort_keys=True, indent=1)
print("      live request returned", sum(len(r) for r in payload["result"]), "documents")
PYEOF
kill $LIVE; wait $LIVE 2>/dev/null || true

echo "[3/5] replay the snapshot with no index and no network"
PYTHONPATH=src "$PY" scripts/serve_search.py --replay-snapshot "$WORK/live.jsonl" \
  --port 8078 --snapshot "$WORK/replay.jsonl" > "$WORK/serve_replay.log" 2>&1 &
REPLAY=$!
trap 'kill $REPLAY 2>/dev/null || true' EXIT
for _ in $(seq 30); do grep -q listening "$WORK/serve_replay.log" && break; sleep 1; done

"$PY" - "$QUERIES" "$WORK/replay_results.json" <<'PYEOF'
import json, sys, urllib.request
queries = json.loads(sys.argv[1])
body = json.dumps({"queries": queries, "topk": 3}).encode()
req = urllib.request.Request("http://127.0.0.1:8078/retrieve", data=body,
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=60) as response:
    payload = json.load(response)
json.dump(payload, open(sys.argv[2], "w"), sort_keys=True, indent=1)
PYEOF

echo "[4/5] compare live and replayed evidence"
"$PY" - "$WORK/live_results.json" "$WORK/replay_results.json" <<'PYEOF'
import hashlib, json, sys


def contents(path):
    payload = json.load(open(path))
    return [[hit["document"]["contents"] for hit in result] for result in payload["result"]]


live, replay = contents(sys.argv[1]), contents(sys.argv[2])
same = live == replay
fingerprint = hashlib.sha256(json.dumps(live, sort_keys=True).encode()).hexdigest()[:16]
print("      evidence sha256:", fingerprint)
print("      replay identical:", same)
if not same:
    raise SystemExit("replay diverged from the live run")
PYEOF

echo "[5/5] an unseen query must fail rather than reach the network"
"$PY" - <<'PYEOF'
import json, urllib.error, urllib.request
body = json.dumps({"queries": ["a query never served live"], "topk": 3}).encode()
req = urllib.request.Request("http://127.0.0.1:8078/retrieve", data=body,
                             headers={"Content-Type": "application/json"})
try:
    urllib.request.urlopen(req, timeout=30)
except urllib.error.HTTPError as error:
    print("      refused with HTTP", error.code, "as expected")
else:
    raise SystemExit("replay backend answered a query absent from the snapshot")
PYEOF
echo
echo "replay equivalence demonstrated; artifacts under $WORK"
