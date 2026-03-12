from __future__ import annotations

import logging
import shutil
from datetime import datetime

import torch


RANK_PTS = [90, 45, 0, -135]


def write_train_metrics(
    *,
    writer,
    stats: dict,
    save_every: int,
    scheduler,
    all_q: torch.Tensor,
    all_q_target: torch.Tensor,
    steps: int,
    online: bool,
) -> None:
    # downsample to reduce tensorboard event size
    all_q_1d = all_q.cpu().numpy().flatten()[::128]
    all_q_target_1d = all_q_target.cpu().numpy().flatten()[::128]

    writer.add_scalar("loss/dqn_loss", stats["dqn_loss"] / save_every, steps)
    if not online:
        writer.add_scalar("loss/cql_loss", stats["cql_loss"] / save_every, steps)
    writer.add_scalar("loss/next_rank_loss", stats["next_rank_loss"] / save_every, steps)
    writer.add_scalar("hparam/lr", scheduler.get_last_lr()[0], steps)
    writer.add_histogram("q_predicted", all_q_1d, steps)
    writer.add_histogram("q_target", all_q_target_1d, steps)
    writer.flush()


def reset_loss_stats(stats: dict) -> None:
    for key in stats:
        stats[key] = 0


def log_total_steps(*, test_every: int, steps: int) -> None:
    before_next_test_play = (test_every - steps % test_every) % test_every
    logging.info(f"total steps: {steps:,} (~{before_next_test_play:,})")


def build_training_state(
    *,
    mortal,
    dqn,
    aux_net,
    optimizer,
    scheduler,
    scaler,
    steps: int,
    best_perf: dict,
    config: dict,
) -> dict:
    return {
        "mortal": mortal.state_dict(),
        "current_dqn": dqn.state_dict(),
        "aux_net": aux_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "steps": steps,
        "timestamp": datetime.now().timestamp(),
        "best_perf": best_perf,
        "config": config,
    }


def save_training_state(*, state: dict, state_file: str) -> None:
    torch.save(state, state_file)


def run_test_play_evaluation(
    *,
    test_player,
    writer,
    mortal,
    dqn,
    device,
    test_games: int,
    steps: int,
    best_perf: dict,
) -> tuple[dict, bool, dict | None]:
    stat = test_player.test_play(test_games // 4, mortal, dqn, device)
    mortal.train()
    dqn.train()

    avg_pt = stat.avg_pt(RANK_PTS)  # for display only, never used in training
    better = avg_pt >= best_perf["avg_pt"] and stat.avg_rank <= best_perf["avg_rank"]
    past_best = best_perf.copy() if better else None
    if better:
        best_perf["avg_pt"] = avg_pt
        best_perf["avg_rank"] = stat.avg_rank

    logging.info(f"avg rank: {stat.avg_rank:.6}")
    logging.info(f"avg pt: {avg_pt:.6}")
    writer.add_scalar("test_play/avg_ranking", stat.avg_rank, steps)
    writer.add_scalar("test_play/avg_pt", avg_pt, steps)
    writer.add_scalars(
        "test_play/ranking",
        {
            "1st": stat.rank_1_rate,
            "2nd": stat.rank_2_rate,
            "3rd": stat.rank_3_rate,
            "4th": stat.rank_4_rate,
        },
        steps,
    )
    writer.add_scalars(
        "test_play/behavior",
        {
            "agari": stat.agari_rate,
            "houjuu": stat.houjuu_rate,
            "fuuro": stat.fuuro_rate,
            "riichi": stat.riichi_rate,
        },
        steps,
    )
    writer.add_scalars(
        "test_play/agari_point",
        {
            "overall": stat.avg_point_per_agari,
            "riichi": stat.avg_point_per_riichi_agari,
            "fuuro": stat.avg_point_per_fuuro_agari,
            "dama": stat.avg_point_per_dama_agari,
        },
        steps,
    )
    writer.add_scalar("test_play/houjuu_point", stat.avg_point_per_houjuu, steps)
    writer.add_scalar("test_play/point_per_round", stat.avg_point_per_round, steps)
    writer.add_scalars(
        "test_play/key_step",
        {
            "agari_jun": stat.avg_agari_jun,
            "houjuu_jun": stat.avg_houjuu_jun,
            "riichi_jun": stat.avg_riichi_jun,
        },
        steps,
    )
    writer.add_scalars(
        "test_play/riichi",
        {
            "agari_after_riichi": stat.agari_rate_after_riichi,
            "houjuu_after_riichi": stat.houjuu_rate_after_riichi,
            "chasing_riichi": stat.chasing_riichi_rate,
            "riichi_chased": stat.riichi_chased_rate,
        },
        steps,
    )
    writer.add_scalar("test_play/riichi_point", stat.avg_riichi_point, steps)
    writer.add_scalars(
        "test_play/fuuro",
        {
            "agari_after_fuuro": stat.agari_rate_after_fuuro,
            "houjuu_after_fuuro": stat.houjuu_rate_after_fuuro,
        },
        steps,
    )
    writer.add_scalar("test_play/fuuro_num", stat.avg_fuuro_num, steps)
    writer.add_scalar("test_play/fuuro_point", stat.avg_fuuro_point, steps)
    writer.flush()
    return best_perf, better, past_best


def save_best_checkpoint(
    *,
    state: dict,
    state_file: str,
    best_state_file: str,
    best_perf: dict,
    past_best: dict,
) -> None:
    save_training_state(state=state, state_file=state_file)
    logging.info(
        "a new record has been made, "
        f"pt: {past_best['avg_pt']:.4} -> {best_perf['avg_pt']:.4}, "
        f"rank: {past_best['avg_rank']:.4} -> {best_perf['avg_rank']:.4}, "
        f"saving to {best_state_file}"
    )
    shutil.copy(state_file, best_state_file)
