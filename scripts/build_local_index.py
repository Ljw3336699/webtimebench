#!/usr/bin/env python3
"""Build a frozen SQLite BM25 index from source-evidence JSONL, without API calls."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from timeagent.search.local_backend import build_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = build_index(args.corpus, args.output)
    except (ValueError, OSError) as error:
        parser.exit(2, f"build_local_index: {error}\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
