from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import toml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = ROOT / ".utils" / "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reproducible Mortal training experiment run directory."
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Experiment name. A timestamp prefix is added automatically.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        required=True,
        help="Base TOML config file to clone and modify.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help="Parent directory for experiment runs.",
    )
    parser.add_argument(
        "--shared-file-index",
        type=Path,
        help="Optional shared dataset.file_index path to reuse across experiments.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override in dotted form, e.g. reward.type=hybrid. Repeatable.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Launch training immediately after preparing the run directory.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use when --run is set.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reuse an existing run directory if it already exists.",
    )
    return parser.parse_args()


def parse_override(raw: str) -> tuple[str, object]:
    if "=" not in raw:
        raise ValueError(f"override must contain '=': {raw}")
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"override key is empty: {raw}")
    return key, parse_value(value.strip())


def parse_value(raw: str):
    lower = raw.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower == "null":
        return None

    try:
        return int(raw)
    except ValueError:
        pass

    try:
        return float(raw)
    except ValueError:
        pass

    if (raw.startswith("[") and raw.endswith("]")) or (raw.startswith("{") and raw.endswith("}")):
        return json.loads(raw)

    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]

    return raw


def set_nested(config: dict, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    cursor = config
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def sanitize_name(name: str) -> str:
    keep = []
    for ch in name.strip():
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    sanitized = "".join(keep).strip("_")
    return sanitized or "run"


def build_run_dir(run_root: Path, experiment_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return run_root / f"{timestamp}_{sanitize_name(experiment_name)}"


def apply_default_run_paths(config: dict, run_dir: Path, shared_file_index: Path | None) -> None:
    checkpoints_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    indices_dir = run_dir / "indices"
    logs_dir = run_dir / "logs"
    online_dir = run_dir / "online"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    indices_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    online_dir.mkdir(parents=True, exist_ok=True)

    set_nested(config, "control.state_file", str(checkpoints_dir / "mortal.pth"))
    set_nested(config, "control.best_state_file", str(checkpoints_dir / "best.pth"))
    set_nested(config, "control.tensorboard_dir", str(tensorboard_dir))
    if shared_file_index is None:
        set_nested(config, "dataset.file_index", str(indices_dir / "file_index.pth"))
    else:
        set_nested(config, "dataset.file_index", str(shared_file_index.resolve()))

    if "test_play" in config:
        set_nested(config, "test_play.log_dir", str(logs_dir / "test_play"))
    if "train_play" in config and "default" in config["train_play"]:
        set_nested(config, "train_play.default.log_dir", str(logs_dir / "train_play_default"))
    if "online" in config and "server" in config["online"]:
        set_nested(config, "online.server.buffer_dir", str(online_dir / "buffer"))
        set_nested(config, "online.server.drain_dir", str(online_dir / "drain"))
    if "1v3" in config:
        set_nested(config, "1v3.log_dir", str(logs_dir / "one_vs_three"))


def write_metadata(
    *,
    run_dir: Path,
    base_config: Path,
    config_path: Path,
    overrides: dict[str, object],
    command: list[str],
) -> None:
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_config": str(base_config.resolve()),
        "config_path": str(config_path.resolve()),
        "cwd": str(ROOT),
        "command": command,
        "overrides": overrides,
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()

    base_config = args.base_config.resolve()
    if not base_config.exists():
        raise SystemExit(f"base config not found: {base_config}")

    with open(base_config, encoding="utf-8") as handle:
        config = toml.load(handle)

    run_root = args.run_root.resolve()
    run_dir = build_run_dir(run_root, args.name)
    if run_dir.exists() and not args.force:
        raise SystemExit(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    apply_default_run_paths(config, run_dir, args.shared_file_index)

    parsed_overrides: dict[str, object] = {}
    for raw_override in args.overrides:
        key, value = parse_override(raw_override)
        parsed_overrides[key] = value
        set_nested(config, key, value)

    config_path = run_dir / "config.toml"
    with open(config_path, "w", encoding="utf-8") as handle:
        toml.dump(config, handle)

    command = [args.python, str(ROOT / "mortal" / "train.py")]
    write_metadata(
        run_dir=run_dir,
        base_config=base_config,
        config_path=config_path,
        overrides=parsed_overrides,
        command=command,
    )

    launch_env = os.environ.copy()
    launch_env["MORTAL_CFG"] = str(config_path)

    print(f"Prepared run directory: {run_dir}")
    print(f"Config: {config_path}")
    print(f"MORTAL_CFG={config_path}")
    print(f"Run command: {' '.join(command)}")

    if args.run:
        subprocess.run(
            command,
            cwd=ROOT / "mortal",
            env=launch_env,
            check=True,
        )


if __name__ == "__main__":
    main()
