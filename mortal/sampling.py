from __future__ import annotations

import random


ACTION_BUCKETS = (
    "tile_select",
    "pass",
    "riichi",
    "chi_low",
    "chi_mid",
    "chi_high",
    "pon",
    "kan",
    "agari",
    "ryukyoku",
)


def action_bucket(action: int) -> str:
    if 0 <= action <= 36:
        return "tile_select"
    return {
        37: "riichi",
        38: "chi_low",
        39: "chi_mid",
        40: "chi_high",
        41: "pon",
        42: "kan",
        43: "agari",
        44: "ryukyoku",
        45: "pass",
    }.get(action, "tile_select")


def extract_action(entry, oracle: bool) -> int:
    action_idx = 2 if oracle else 1
    return int(entry[action_idx])


def count_action_buckets(actions) -> dict[str, int]:
    counts = {bucket: 0 for bucket in ACTION_BUCKETS}
    for action in actions.tolist() if hasattr(actions, "tolist") else actions:
        counts[action_bucket(int(action))] += 1
    return counts


class UniformSampler:
    name = "uniform"

    def resample_buffer(self, buffer):
        return buffer


class ActionBucketSampler:
    name = "action_bucket"

    def __init__(self, *, oracle: bool, bucket_weights: dict[str, float]):
        self.oracle = oracle
        self.bucket_weights = bucket_weights

    def resample_buffer(self, buffer):
        if len(buffer) <= 1:
            return buffer

        weights = []
        for entry in buffer:
            bucket = action_bucket(extract_action(entry, self.oracle))
            weight = self.bucket_weights.get(bucket, 1.0)
            weights.append(max(float(weight), 0.0))

        if max(weights, default=0.0) <= 0.0:
            return buffer
        if all(weight == 1.0 for weight in weights):
            return buffer

        return random.choices(buffer, weights=weights, k=len(buffer))


def build_sampler(config, oracle: bool):
    sampling_cfg = config.get("sampling", {})
    sampling_type = sampling_cfg.get("type", "uniform")

    if sampling_type == "uniform":
        return UniformSampler()

    if sampling_type == "action_bucket":
        raw_bucket_weights = sampling_cfg.get("bucket_weights", {})
        bucket_weights = {bucket: 1.0 for bucket in ACTION_BUCKETS}
        for bucket, value in raw_bucket_weights.items():
            if bucket not in bucket_weights:
                raise ValueError(f"unknown sampling bucket: {bucket}")
            bucket_weights[bucket] = float(value)
        return ActionBucketSampler(oracle=oracle, bucket_weights=bucket_weights)

    raise ValueError(f"unknown sampling.type: {sampling_type}")
