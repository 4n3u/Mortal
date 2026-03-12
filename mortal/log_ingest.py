from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class LogIngestError(Exception):
    pass


class UnsupportedLogSourceError(LogIngestError):
    pass


@dataclass(frozen=True)
class LogMetadata:
    source: str
    event_count: int
    has_empty_dora_marker: bool


def open_log_text(path: str | Path) -> str:
    path = Path(path)
    with open(path, "rb") as handle:
        prefix = handle.read(2)
    if path.suffix.lower() == ".gz" or prefix == b"\x1f\x8b":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8")


def parse_json_lines(raw_log: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in raw_log.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        events.append(json.loads(stripped))
    return events


def detect_log_source(path: str | Path | None, events: list[dict[str, Any]]) -> str:
    if path is not None:
        lower_parts = [part.lower() for part in Path(path).parts]
        if "tenhou" in lower_parts:
            return "tenhou"
        if "majsoul" in lower_parts:
            return "majsoul"

    if not events:
        return "unknown"

    first = events[0]
    if "aka_flag" in first or "kyoku_first" in first:
        return "tenhou"
    if any(
        event.get("type") == "start_kyoku" and event.get("dora_marker", None) == ""
        for event in events[:10]
    ):
        return "majsoul"
    return "unknown"


def collect_log_metadata(path: str | Path | None, events: list[dict[str, Any]]) -> LogMetadata:
    has_empty_dora_marker = any(
        event.get("type") == "start_kyoku" and event.get("dora_marker", None) == ""
        for event in events
    )
    return LogMetadata(
        source=detect_log_source(path, events),
        event_count=len(events),
        has_empty_dora_marker=has_empty_dora_marker,
    )


def normalize_for_gameplay_loader(
    raw_log: str,
    *,
    path: str | Path | None = None,
) -> tuple[str, LogMetadata]:
    events = parse_json_lines(raw_log)
    meta = collect_log_metadata(path, events)

    if meta.source == "majsoul" and meta.has_empty_dora_marker:
        raise UnsupportedLogSourceError(
            "majsoul log missing start_kyoku.dora_marker; normalization rule is not implemented"
        )

    normalized = "\n".join(json.dumps(event, ensure_ascii=False) for event in events)
    if normalized:
        normalized += "\n"
    return normalized, meta
