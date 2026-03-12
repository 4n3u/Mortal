def train():
    import prelude

    import logging
    import sys
    import os
    import gc
    import shutil
    import random
    import torch
    from os import path
    from datetime import datetime
    from itertools import chain
    from torch import optim
    from torch.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from common import submit_param, parameter_count, drain, filtered_trimmed_lines, tqdm
    from player import TestPlayer
    from dataloader import FileDatasetsIter, worker_init_fn
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from model import Brain, DQN, AuxNet
    from oracle_guiding import OracleGuidingConfig
    from sampling import ACTION_BUCKETS, count_action_buckets
    from training_data import build_offline_file_list, load_player_names
    from training_hooks import (
        build_training_state,
        log_total_steps,
        reset_loss_stats,
        run_test_play_evaluation,
        save_best_checkpoint,
        save_training_state,
        write_train_metrics,
    )
    from training_losses import TrainingLossComputer, accumulate_loss_stats
    from libriichi.consts import obs_shape
    from config import config

    version = config['control']['version']

    online = config['control']['online']
    batch_size = config['control']['batch_size']
    opt_step_every = config['control']['opt_step_every']
    save_every = config['control']['save_every']
    test_every = config['control']['test_every']
    submit_every = config['control']['submit_every']
    test_games = config['test_play']['games']
    min_q_weight = config['cql']['min_q_weight']
    next_rank_weight = config['aux']['next_rank_weight']
    oracle_guiding = OracleGuidingConfig.from_config(config)
    assert save_every % opt_step_every == 0
    assert test_every % save_every == 0

    device = torch.device(config['control']['device'])
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
    enable_amp = config['control']['enable_amp']
    enable_compile = config['control']['enable_compile']

    pts = config['env']['pts']
    gamma = config['env']['gamma']
    file_batch_size = config['dataset']['file_batch_size']
    reserve_ratio = config['dataset']['reserve_ratio']
    num_workers = config['dataset']['num_workers']
    num_epochs = config['dataset']['num_epochs']
    enable_augmentation = config['dataset']['enable_augmentation']
    augmented_first = config['dataset']['augmented_first']
    eps = config['optim']['eps']
    betas = config['optim']['betas']
    weight_decay = config['optim']['weight_decay']
    max_grad_norm = config['optim']['max_grad_norm']

    mortal = Brain(version=version, **config['resnet']).to(device)
    dqn = DQN(version=version).to(device)
    aux_net = AuxNet((4,)).to(device)
    oracle_mortal = None
    if oracle_guiding.enabled:
        oracle_mortal = Brain(version=version, is_oracle=True, **config['resnet']).to(device)

    all_models = (mortal, dqn, aux_net) if oracle_mortal is None else (mortal, dqn, aux_net, oracle_mortal)
    if enable_compile:
        for m in all_models:
            m.compile()

    logging.info(f'version: {version}')
    logging.info(f'obs shape: {obs_shape(version)}')
    logging.info(f'mortal params: {parameter_count(mortal):,}')
    logging.info(f'dqn params: {parameter_count(dqn):,}')
    logging.info(f'aux params: {parameter_count(aux_net):,}')
    if oracle_mortal is not None:
        logging.info(f'oracle mortal params: {parameter_count(oracle_mortal):,}')

    mortal.freeze_bn(config['freeze_bn']['mortal'])
    if oracle_mortal is not None:
        oracle_mortal.freeze_bn(config['freeze_bn']['mortal'])

    decay_params = []
    no_decay_params = []
    for model in all_models:
        params_dict = {}
        to_decay = set()
        for mod_name, mod in model.named_modules():
            for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
                params_dict[name] = param
                if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                    to_decay.add(name)
        decay_params.extend(params_dict[name] for name in sorted(to_decay))
        no_decay_params.extend(params_dict[name] for name in sorted(params_dict.keys() - to_decay))
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params},
    ]
    optimizer = optim.AdamW(param_groups, lr=1, weight_decay=0, betas=betas, eps=eps)
    scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **config['optim']['scheduler'])
    scaler = GradScaler(device.type, enabled=enable_amp)
    test_player = TestPlayer()
    best_perf = {
        'avg_rank': 4.,
        'avg_pt': -135.,
    }

    steps = 0
    state_file = config['control']['state_file']
    best_state_file = config['control']['best_state_file']
    if path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f'loaded: {timestamp}')
        mortal.load_state_dict(state['mortal'])
        dqn.load_state_dict(state['current_dqn'])
        aux_net.load_state_dict(state['aux_net'])
        if not online or state['config']['control']['online']:
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        best_perf = state['best_perf']
        steps = state['steps']
        if oracle_mortal is not None and 'oracle_mortal' in state:
            oracle_mortal.load_state_dict(state['oracle_mortal'])

    optimizer.zero_grad(set_to_none=True)
    loss_computer = TrainingLossComputer(
        min_q_weight = min_q_weight,
        next_rank_weight = next_rank_weight,
        online = online,
        oracle_guiding = oracle_guiding,
    )

    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')

    if online:
        submit_param(mortal, dqn, is_idle=True)
        logging.info('param has been submitted')

    writer = SummaryWriter(config['control']['tensorboard_dir'])
    stats = {
        'dqn_loss': 0,
        'cql_loss': 0,
        'next_rank_loss': 0,
    }
    if oracle_guiding.enabled:
        stats.update({
            'oracle_dqn_loss': 0,
            'oracle_aux_loss': 0,
            'oracle_alignment_loss': 0,
        })
    all_q = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
    all_q_target = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
    action_bucket_counts = {bucket: 0 for bucket in ACTION_BUCKETS}
    idx = 0

    def train_epoch():
        nonlocal steps
        nonlocal idx

        player_names = []
        if online:
            player_names = ['trainee']
            dirname = drain()
            file_list = list(map(lambda p: path.join(dirname, p), os.listdir(dirname)))
        else:
            player_names = load_player_names(config['dataset']['player_names_files'])
            file_list = build_offline_file_list(config['dataset'], player_names)
        logging.info(f'file list size: {len(file_list):,}')

        log_total_steps(test_every = test_every, steps = steps)

        if num_workers > 1:
            random.shuffle(file_list)
        file_data = FileDatasetsIter(
            version = version,
            file_list = file_list,
            pts = pts,
            oracle = oracle_guiding.enabled,
            file_batch_size = file_batch_size,
            reserve_ratio = reserve_ratio,
            player_names = player_names,
            num_epochs = num_epochs,
            enable_augmentation = enable_augmentation,
            augmented_first = augmented_first,
        )
        data_loader = iter(DataLoader(
            dataset = file_data,
            batch_size = batch_size,
            drop_last = False,
            num_workers = num_workers,
            pin_memory = True,
            worker_init_fn = worker_init_fn,
        ))

        remaining_obs = []
        remaining_invisible_obs = []
        remaining_actions = []
        remaining_masks = []
        remaining_steps_to_done = []
        remaining_kyoku_rewards = []
        remaining_player_ranks = []
        remaining_bs = 0
        pb = tqdm(total=save_every, desc='TRAIN', initial=steps % save_every)

        def train_batch(obs, invisible_obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks):
            nonlocal steps
            nonlocal idx
            nonlocal pb

            obs = obs.to(dtype=torch.float32, device=device)
            if invisible_obs is not None:
                invisible_obs = invisible_obs.to(dtype=torch.float32, device=device)
            batch_action_bucket_counts = count_action_buckets(actions)
            actions = actions.to(dtype=torch.int64, device=device)
            masks = masks.to(dtype=torch.bool, device=device)
            steps_to_done = steps_to_done.to(dtype=torch.int64, device=device)
            kyoku_rewards = kyoku_rewards.to(dtype=torch.float64, device=device)
            player_ranks = player_ranks.to(dtype=torch.int64, device=device)
            assert masks[range(batch_size), actions].all()

            outputs = loss_computer.compute(
                mortal = mortal,
                dqn = dqn,
                aux_net = aux_net,
                oracle_mortal = oracle_mortal,
                obs = obs,
                invisible_obs = invisible_obs,
                actions = actions,
                masks = masks,
                steps_to_done = steps_to_done,
                kyoku_rewards = kyoku_rewards,
                player_ranks = player_ranks,
                gamma = gamma,
                device = device,
                enable_amp = enable_amp,
            )
            scaler.scale(outputs.loss / opt_step_every).backward()

            with torch.inference_mode():
                accumulate_loss_stats(
                    stats = stats,
                    outputs = outputs,
                    all_q = all_q,
                    all_q_target = all_q_target,
                    idx = idx,
                    online = online,
                )
                for bucket, count in batch_action_bucket_counts.items():
                    action_bucket_counts[bucket] += count

            steps += 1
            idx += 1
            if idx % opt_step_every == 0:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    params = chain.from_iterable(g['params'] for g in optimizer.param_groups)
                    clip_grad_norm_(params, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            pb.update(1)

            if online and steps % submit_every == 0:
                submit_param(mortal, dqn, is_idle=False)
                logging.info('param has been submitted')

            if steps % save_every == 0:
                pb.close()

                write_train_metrics(
                    writer = writer,
                    stats = stats,
                    save_every = save_every,
                    scheduler = scheduler,
                    all_q = all_q,
                    all_q_target = all_q_target,
                    steps = steps,
                    online = online,
                    action_bucket_counts = action_bucket_counts,
                )

                reset_loss_stats(stats)
                for bucket in action_bucket_counts:
                    action_bucket_counts[bucket] = 0
                idx = 0

                log_total_steps(test_every = test_every, steps = steps)

                state = build_training_state(
                    mortal = mortal,
                    dqn = dqn,
                    aux_net = aux_net,
                    optimizer = optimizer,
                    scheduler = scheduler,
                    scaler = scaler,
                    steps = steps,
                    best_perf = best_perf,
                    config = config,
                    extra_state = None if oracle_mortal is None else {'oracle_mortal': oracle_mortal.state_dict()},
                )
                save_training_state(state = state, state_file = state_file)

                if online and steps % submit_every != 0:
                    submit_param(mortal, dqn, is_idle=False)
                    logging.info('param has been submitted')

                if steps % test_every == 0:
                    best_perf, better, past_best = run_test_play_evaluation(
                        test_player = test_player,
                        writer = writer,
                        mortal = mortal,
                        dqn = dqn,
                        device = device,
                        test_games = test_games,
                        steps = steps,
                        best_perf = best_perf,
                    )

                    if better:
                        state['best_perf'] = best_perf
                        save_best_checkpoint(
                            state = state,
                            state_file = state_file,
                            best_state_file = best_state_file,
                            best_perf = best_perf,
                            past_best = past_best,
                        )
                    if online:
                        # BUG: This is a bug with unknown reason. When training
                        # in online mode, the process will get stuck here. This
                        # is the reason why `main` spawns a sub process to train
                        # in online mode instead of going for training directly.
                        sys.exit(0)
                pb = tqdm(total=save_every, desc='TRAIN')

        for batch in data_loader:
            if oracle_guiding.enabled:
                obs, invisible_obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks = batch
            else:
                obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks = batch
                invisible_obs = None
            bs = obs.shape[0]
            if bs != batch_size:
                remaining_obs.append(obs)
                if invisible_obs is not None:
                    remaining_invisible_obs.append(invisible_obs)
                remaining_actions.append(actions)
                remaining_masks.append(masks)
                remaining_steps_to_done.append(steps_to_done)
                remaining_kyoku_rewards.append(kyoku_rewards)
                remaining_player_ranks.append(player_ranks)
                remaining_bs += bs
                continue
            train_batch(obs, invisible_obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks)

        remaining_batches = remaining_bs // batch_size
        if remaining_batches > 0:
            obs = torch.cat(remaining_obs, dim=0)
            invisible_obs = torch.cat(remaining_invisible_obs, dim=0) if remaining_invisible_obs else None
            actions = torch.cat(remaining_actions, dim=0)
            masks = torch.cat(remaining_masks, dim=0)
            steps_to_done = torch.cat(remaining_steps_to_done, dim=0)
            kyoku_rewards = torch.cat(remaining_kyoku_rewards, dim=0)
            player_ranks = torch.cat(remaining_player_ranks, dim=0)
            start = 0
            end = batch_size
            while end <= remaining_bs:
                train_batch(
                    obs[start:end],
                    invisible_obs[start:end] if invisible_obs is not None else None,
                    actions[start:end],
                    masks[start:end],
                    steps_to_done[start:end],
                    kyoku_rewards[start:end],
                    player_ranks[start:end],
                )
                start = end
                end += batch_size
        pb.close()

        if online:
            submit_param(mortal, dqn, is_idle=True)
            logging.info('param has been submitted')

    while True:
        train_epoch()
        gc.collect()
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        if not online:
            # only run one epoch for offline for easier control
            break

def main():
    import os
    import sys
    import time
    from subprocess import Popen
    from config import config

    # do not set this env manually
    is_sub_proc_key = 'MORTAL_IS_SUB_PROC'
    online = config['control']['online']
    if not online or os.environ.get(is_sub_proc_key, '0') == '1':
        train()
        return

    cmd = (sys.executable, __file__)
    env = {
        is_sub_proc_key: '1',
        **os.environ.copy(),
    }
    while True:
        child = Popen(
            cmd,
            stdin = sys.stdin,
            stdout = sys.stdout,
            stderr = sys.stderr,
            env = env,
        )
        if (code := child.wait()) != 0:
            sys.exit(code)
        time.sleep(3)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
