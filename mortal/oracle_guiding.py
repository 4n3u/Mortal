from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OracleGuidingConfig:
    enabled: bool = False
    oracle_dqn_weight: float = 1.0
    oracle_aux_weight: float = 0.2
    alignment_weight: float = 0.05
    detach_target: bool = True

    @classmethod
    def from_config(cls, config: dict) -> "OracleGuidingConfig":
        cfg = config.get("oracle_guiding", {})
        return cls(
            enabled = cfg.get("enabled", False),
            oracle_dqn_weight = cfg.get("oracle_dqn_weight", 1.0),
            oracle_aux_weight = cfg.get("oracle_aux_weight", 0.2),
            alignment_weight = cfg.get("alignment_weight", 0.05),
            detach_target = cfg.get("detach_target", True),
        )
