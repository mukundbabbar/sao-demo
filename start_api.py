#!/usr/bin/env python3
"""Start the banking /query API with plain uvicorn (no OTel auto-instrumentor).

Session/project ids are stamped on the Splunk AO agent trace in api.py.

Usage:
    python start_api.py
    PORT=8000 python start_api.py
"""
from __future__ import annotations

import os
import shutil
import sys


def main() -> None:
    uvicorn = shutil.which("uvicorn")
    if not uvicorn:
        sys.exit(
            "uvicorn not found. Install deps with "
            "`pip install -r requirements.txt` (needs Python 3.11+)."
        )
    port = os.environ.get("PORT", "8000")
    os.execvp(
        uvicorn,
        [
            "uvicorn",
            "api:app",
            "--host",
            "0.0.0.0",
            "--port",
            port,
        ],
    )


if __name__ == "__main__":
    main()
