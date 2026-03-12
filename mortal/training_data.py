from __future__ import annotations

import gzip
import hashlib
import json
import logging
from glob import glob
from os import path
from pathlib import Path

import torch
from functools import partial
from tqdm.auto import tqdm as orig_tqdm

tqdm = partial(orig_tqdm, unit="batch", dynamic_ncols=True, ascii=True)


SUPPORTED_SOURCES = {"tenhou", "majsoul", "unknown"}
INDEX_VERSION = 1


def load_player_names(player_name_files: list[str]) -> list[str]:
    player_names_set: set[str] = set()
    for filename in player_name_files:
        with open(filename, encoding="utf-8") as handle:
            player_names_set.update(filtered_trimmed_lines(handle))
    player_names = sorted(player_names_set)
    logging.info(f"loaded {len(player_names):,} players")
    return player_names


def build_offline_file_list(dataset_cfg: dict, player_names: list[str]) -> list[str]:
    file_index = dataset_cfg["file_index"]
    max_files = int(dataset_cfg.get("max_files", 0))
    meta = {
        "index_version": INDEX_VERSION,
        "globs": list(dataset_cfg["globs"]),
        "sources": normalize_sources(dataset_cfg.get("sources")),
        "player_names_fingerprint": fingerprint_player_names(player_names),
        "max_files": max_files,
    }

    if path.exists(file_index):
        index = torch.load(file_index, weights_only=True)
        if index.get("meta") == meta:
            return index["file_list"]

        logging.info("dataset index metadata changed, rebuilding file index...")

    logging.info("building file index...")
    file_list = []
    for pat in dataset_cfg["globs"]:
        file_list.extend(glob(pat, recursive=True))

    source_counts: dict[str, int] = {}
    allowed_sources = set(meta["sources"])
    if allowed_sources:
        filtered = []
        for filename in tqdm(file_list, unit="file"):
            first_event = read_first_event(filename)
            source = detect_source(filename, first_event)
            source_counts[source] = source_counts.get(source, 0) + 1
            if source in allowed_sources:
                filtered.append(filename)
        file_list = filtered
        logging.info(f"source-filtered file list size: {len(file_list):,}")
        logging.info(f"source counts before filtering: {source_counts}")

    if player_names:
        player_names_set = set(player_names)
        filtered = []
        for filename in tqdm(file_list, unit="file"):
            first_event = read_first_event(filename)
            if not set(first_event.get("names", [])).isdisjoint(player_names_set):
                filtered.append(filename)
        file_list = filtered

    file_list.sort(reverse=True)
    if max_files > 0:
        file_list = file_list[:max_files]
        logging.info(f"truncated file list to max_files={max_files:,}")
    torch.save({"meta": meta, "file_list": file_list}, file_index)
    return file_list


def normalize_sources(raw_sources) -> list[str]:
    if raw_sources is None:
        return ["tenhou"]
    if isinstance(raw_sources, str):
        raw_sources = [raw_sources]

    sources = [str(source).strip().lower() for source in raw_sources if str(source).strip()]
    invalid = sorted(set(sources) - SUPPORTED_SOURCES)
    if invalid:
        raise ValueError(f"unsupported dataset sources: {invalid}")
    return sources


def fingerprint_player_names(player_names: list[str]) -> str:
    joined = "\n".join(sorted(player_names))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def filtered_trimmed_lines(lines) -> list[str]:
    return list(filter(lambda line: line, map(lambda line: line.strip(), lines)))


def read_first_event(filename: str) -> dict:
    opener = open
    mode = "rt"
    kwargs = {"encoding": "utf-8"}
    if is_gzip_file(filename):
        opener = gzip.open

    with opener(filename, mode, **kwargs) as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                return json.loads(stripped)
    return {}


def is_gzip_file(filename: str) -> bool:
    with open(filename, "rb") as handle:
        prefix = handle.read(2)
    return prefix == b"\x1f\x8b" or Path(filename).suffix.lower() == ".gz"


def detect_source(filename: str, first_event: dict) -> str:
    lower_parts = [part.lower() for part in Path(filename).parts]
    if "tenhou" in lower_parts:
        return "tenhou"
    if "majsoul" in lower_parts:
        return "majsoul"
    if "aka_flag" in first_event or "kyoku_first" in first_event:
        return "tenhou"
    if first_event.get("type") == "start_kyoku" and first_event.get("dora_marker", None) == "":
        return "majsoul"
    return "unknown"
