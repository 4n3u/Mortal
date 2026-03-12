from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple Mortal experiment configs sequentially."
    )
    parser.add_argument(
        "--config",
        dest="configs",
        action="append",
        required=True,
        help="Path to an experiment config.toml. Repeatable in desired run order.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with the next experiment if one run fails.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use.",
    )
    return parser.parse_args()


def run_one(config_path: Path, python_exe: str) -> int:
    run_dir = config_path.parent
    log_path = run_dir / "train.log"
    status_path = run_dir / "status.json"

    env = os.environ.copy()
    env["MORTAL_CFG"] = str(config_path)
    command = [python_exe, str(ROOT / "mortal" / "train.py")]

    status = {
        "config": str(config_path),
        "run_dir": str(run_dir),
        "command": command,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    with open(status_path, "w", encoding="utf-8") as handle:
        json.dump(status, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"\n=== START {status['started_at']} ===\n")
        handle.flush()
        proc = subprocess.run(
            command,
            cwd=ROOT / "mortal",
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

    status["ended_at"] = datetime.now(timezone.utc).isoformat()
    status["return_code"] = proc.returncode
    status["status"] = "completed" if proc.returncode == 0 else "failed"
    with open(status_path, "w", encoding="utf-8") as handle:
        json.dump(status, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return proc.returncode


def main() -> None:
    args = parse_args()
    configs = [Path(config).resolve() for config in args.configs]

    for config in configs:
        if not config.exists():
            raise SystemExit(f"config not found: {config}")

    for config in configs:
        return_code = run_one(config, args.python)
        print(f"{config}: return_code={return_code}")
        if return_code != 0 and not args.continue_on_error:
            raise SystemExit(return_code)


if __name__ == "__main__":
    main()
