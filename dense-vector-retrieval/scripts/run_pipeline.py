#!/usr/bin/env python3
"""validate -> encode -> retrieve -> evaluate. Agent should call this instead of rewriting pooling."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def run_stage(script: str, config: str, work_dir: str | None) -> None:
    cmd = [sys.executable, str(SCRIPTS / script), "--config", config]
    if work_dir is not None:
        cmd.extend(["--work-dir", work_dir])
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "validate", "encode", "retrieve", "evaluate"],
    )
    args = parser.parse_args()

    order = {
        "validate": [("validate_config.py", False)],
        "encode": [("encode.py", True)],
        "retrieve": [("retrieve.py", True)],
        "evaluate": [("evaluate.py", True)],
        "all": [
            ("validate_config.py", False),
            ("encode.py", True),
            ("retrieve.py", True),
            ("evaluate.py", True),
        ],
    }
    for script, needs_work in order[args.stage]:
        run_stage(script, args.config, args.work_dir if needs_work else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
