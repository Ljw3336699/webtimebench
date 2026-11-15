#!/usr/bin/env python3
"""Run the revised search endpoint without installing the GPU training stack."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from timeagent.search.web_server import main

if __name__ == "__main__":
    main()
