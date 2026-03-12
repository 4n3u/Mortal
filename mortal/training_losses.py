from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LossBatchOutputs:
    loss: torch.Tensor
    q: torch.Tensor
    q_target_mc: torch.Tensor
    dqn_loss: torch.Tensor
    cql_loss: torch.Tensor
    next_rank_loss: torch.Tensor


class TrainingLossComputer:
    def __init__(self, *, min_q_weight: float, next_rank_weight: float, online: bool):
        self.min_q_weight = min_q_weight
        self.next_rank_weight = next_rank_weight
        self.online = online
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def compute(
        self,
        *,
        mortal,
        dqn,
        aux_net,
        obs: torch.Tensor,
        actions: torch.Tensor,
        masks: torch.Tensor,
        steps_to_done: torch.Tensor,
        kyoku_rewards: torch.Tensor,
        player_ranks: torch.Tensor,
        gamma: float,
        device: torch.device,
        enable_amp: bool,
    ) -> LossBatchOutputs:
        batch_size = actions.shape[0]
        batch_indices = torch.arange(batch_size, device=device)

        q_target_mc = gamma ** steps_to_done * kyoku_rewards
        q_target_mc = q_target_mc.to(torch.float32)

        with torch.autocast(device.type, enabled=enable_amp):
            phi = mortal(obs)
            q_out = dqn(phi, masks)
            q = q_out[batch_indices, actions]
            dqn_loss = 0.5 * self.mse(q, q_target_mc)

            cql_loss = torch.zeros((), device=device, dtype=q.dtype)
            if not self.online:
                cql_loss = q_out.logsumexp(-1).mean() - q.mean()

            next_rank_logits, = aux_net(phi)
            next_rank_loss = self.ce(next_rank_logits, player_ranks)

            loss = sum((
                dqn_loss,
                cql_loss * self.min_q_weight,
                next_rank_loss * self.next_rank_weight,
            ))

        return LossBatchOutputs(
            loss = loss,
            q = q,
            q_target_mc = q_target_mc,
            dqn_loss = dqn_loss,
            cql_loss = cql_loss,
            next_rank_loss = next_rank_loss,
        )


def accumulate_loss_stats(
    *,
    stats: dict,
    outputs: LossBatchOutputs,
    all_q: torch.Tensor,
    all_q_target: torch.Tensor,
    idx: int,
    online: bool,
) -> None:
    stats["dqn_loss"] += outputs.dqn_loss
    if not online:
        stats["cql_loss"] += outputs.cql_loss
    stats["next_rank_loss"] += outputs.next_rank_loss
    all_q[idx] = outputs.q
    all_q_target[idx] = outputs.q_target_mc
