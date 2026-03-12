from __future__ import annotations

import numpy as np
import torch

from model import GRP
from reward_calculator import RewardCalculator


def normalize_rank_by_player(value) -> np.ndarray:
    if isinstance(value, (bytes, bytearray)):
        return np.frombuffer(value, dtype=np.uint8).astype(np.int64)
    return np.asarray(value, dtype=np.int64)


def extract_grp_arrays(grp) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grp_feature = np.asarray(grp.take_feature(), dtype=np.float64)
    rank_by_player = normalize_rank_by_player(grp.take_rank_by_player())
    final_scores = np.asarray(grp.take_final_scores(), dtype=np.int64)
    return grp_feature, rank_by_player, final_scores


class BaseRewardProvider:
    name = "base"

    def calc_kyoku_rewards(
        self,
        *,
        player_id: int,
        grp_feature: np.ndarray,
        rank_by_player: np.ndarray,
        final_scores: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError


class GrpDeltaPtRewardProvider(BaseRewardProvider):
    name = "grp_delta_pt"

    def __init__(self, calculator: RewardCalculator):
        self.calculator = calculator

    def calc_kyoku_rewards(
        self,
        *,
        player_id: int,
        grp_feature: np.ndarray,
        rank_by_player: np.ndarray,
        final_scores: np.ndarray,
    ) -> np.ndarray:
        del final_scores
        return np.asarray(
            self.calculator.calc_delta_pt(player_id, grp_feature, rank_by_player),
            dtype=np.float64,
        )


class RawScoreDeltaRewardProvider(BaseRewardProvider):
    name = "raw_score_delta"

    def __init__(self, score_scale: float = 1000.0):
        self.score_scale = float(score_scale)
        if self.score_scale <= 0:
            raise ValueError("score_scale must be > 0")

    def calc_kyoku_rewards(
        self,
        *,
        player_id: int,
        grp_feature: np.ndarray,
        rank_by_player: np.ndarray,
        final_scores: np.ndarray,
    ) -> np.ndarray:
        del rank_by_player
        score_seq = np.concatenate(
            (grp_feature[:, 3 + player_id] * 1e4, [final_scores[player_id]]),
        )
        return np.diff(score_seq) / self.score_scale


class HybridRewardProvider(BaseRewardProvider):
    name = "hybrid"

    def __init__(
        self,
        grp_provider: GrpDeltaPtRewardProvider,
        raw_provider: RawScoreDeltaRewardProvider,
        raw_score_weight: float,
    ):
        if not 0.0 <= raw_score_weight <= 1.0:
            raise ValueError("raw_score_weight must be within [0, 1]")
        self.grp_provider = grp_provider
        self.raw_provider = raw_provider
        self.raw_score_weight = float(raw_score_weight)

    def calc_kyoku_rewards(
        self,
        *,
        player_id: int,
        grp_feature: np.ndarray,
        rank_by_player: np.ndarray,
        final_scores: np.ndarray,
    ) -> np.ndarray:
        grp_reward = self.grp_provider.calc_kyoku_rewards(
            player_id=player_id,
            grp_feature=grp_feature,
            rank_by_player=rank_by_player,
            final_scores=final_scores,
        )
        raw_reward = self.raw_provider.calc_kyoku_rewards(
            player_id=player_id,
            grp_feature=grp_feature,
            rank_by_player=rank_by_player,
            final_scores=final_scores,
        )
        return (
            (1.0 - self.raw_score_weight) * grp_reward
            + self.raw_score_weight * raw_reward
        )


def build_reward_provider(config) -> BaseRewardProvider:
    reward_cfg = config.get("reward", {})
    reward_type = reward_cfg.get("type", "grp_delta_pt")

    if reward_type == "raw_score_delta":
        score_scale = reward_cfg.get("score_scale", 1000.0)
        return RawScoreDeltaRewardProvider(score_scale=score_scale)

    grp_provider = _build_grp_delta_provider(config, reward_cfg)
    if reward_type == "grp_delta_pt":
        return grp_provider

    if reward_type == "hybrid":
        raw_provider = RawScoreDeltaRewardProvider(
            score_scale=reward_cfg.get("score_scale", 1000.0),
        )
        return HybridRewardProvider(
            grp_provider=grp_provider,
            raw_provider=raw_provider,
            raw_score_weight=reward_cfg.get("raw_score_weight", 0.2),
        )

    raise ValueError(f"unknown reward.type: {reward_type}")


def _build_grp_delta_provider(config, reward_cfg) -> GrpDeltaPtRewardProvider:
    grp = GRP(**config["grp"]["network"])
    grp_state = torch.load(
        config["grp"]["state_file"],
        weights_only=True,
        map_location=torch.device("cpu"),
    )
    grp.load_state_dict(grp_state["model"])
    calculator = RewardCalculator(
        grp,
        config["env"]["pts"],
        uniform_init=reward_cfg.get("uniform_init", False),
    )
    return GrpDeltaPtRewardProvider(calculator)
