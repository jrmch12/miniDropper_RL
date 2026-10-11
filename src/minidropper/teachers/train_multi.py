"""RAPTOR pre-training: train one PPO teacher per vehicle config, K at a time.

    python src/minidropper/teachers/train_multi.py num_configs=1000 teachers_per_chunk=128

Outputs (Hydra run dir):
    chunk_XXX.pkl        stacked params of the K teachers + their config rows + eval
    teachers_index.csv   one line per teacher, sorted by eval return, with accept flag
    teachers_meta.json   network config / params table / acceptance thresholds
    metrics.jsonl        per-eval metrics (per-teacher arrays + aggregates)
"""

import csv
import functools
import json
import math
import time
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig, OmegaConf


def network_kwargs(tc):
    return dict(
        policy_dec_hidden_layers=list(tc.policy_dec_hidden_layers),
        action_hidden_layer_sizes=list(tc.action_hidden_layers),
        value_hidden_layer_sizes=list(tc.value_hidden_layers),
        activation=tc.activation,
        policy_obs_key=tc.policy_obs_key,
        value_obs_key=tc.value_obs_key,
    )


@hydra.main(config_name='multi', config_path='./cfg', version_base=None)
def main(cfg: DictConfig):
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.WARNING)
    import jax
    from brax.io import model

    from minidropper.dr import sample_params as sp
    from minidropper.task.dynamics import params_from_dict
    from minidropper.task.rocket import MiniDropperRocket
    from minidropper.task import wrapper
    from minidropper.teachers.training import multi_ppo
    from minidropper.teachers.training import networks as ppo_networks
    from minidropper.utils.paths import resolve
    from minidropper.utils.run_logger import RunLogger

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    tc = cfg.train
    K = cfg.teachers_per_chunk
    E, E_ev = tc.envs_per_teacher, tc.eval_envs_per_teacher

    table = sp.load_table(resolve(cfg.params_table), 'train')
    n_total = len(table['mass'])
    ids_all = np.arange(cfg.config_start, min(cfg.config_start + cfg.num_configs, n_total))
    n_chunks = math.ceil(len(ids_all) / K)
    print(f'[multi] {len(ids_all)} teachers in {n_chunks} chunk(s) of K={K}, '
          f'{K * E} envs, {tc.num_timesteps:,} steps per teacher')

    meta = {
        'kind': 'multi_teacher_ppo',
        'params_table': str(cfg.params_table),
        'network': network_kwargs(tc),
        'env': OmegaConf.to_container(cfg.env, resolve=True),
        'accept_min_return': cfg.accept_min_return,
        'accept_max_terminated': cfg.accept_max_terminated,
        'num_chunks': n_chunks,
    }
    (out_dir / 'teachers_meta.json').write_text(json.dumps(meta, indent=2))

    logger = RunLogger(out_dir, OmegaConf.to_container(cfg, resolve=True), tc.use_wandb,
                       tc.wandb_project, tc.wandb_run_name)
    env = MiniDropperRocket(cfg=cfg.env)
    network_factory = functools.partial(ppo_networks.make_ppo_networks, **network_kwargs(tc))

    rows = []
    t_start = time.time()
    for c in range(n_chunks):
        ids = ids_all[c * K:(c + 1) * K]
        n_real = len(ids)
        ids_padded = np.concatenate([ids, np.repeat(ids[-1:], K - n_real)])  # pad last chunk
        cfg_rows = sp.select(table, ids_padded)
        train_env = wrapper.wrap_for_training(
            env, params_from_dict(sp.repeat_rows(cfg_rows, E)), tc.episode_length)
        eval_env = wrapper.wrap_for_eval(env, params_from_dict(sp.repeat_rows(cfg_rows, E_ev)))

        def progress(step, m, c=c):
            m = dict(m, chunk=c)
            logger.log(c * tc.num_timesteps + step, m,
                       ['eval/return_mean', 'eval/return_min', 'eval/length_mean',
                        'eval/terminated_mean', 'eval/vel_err_mean', 'eval/yaw_err_mean',
                        'train/train_return_mean', 'sps_total'],
                       prefix=f'[chunk {c:03d}] ')

        t_chunk = time.time()
        params, ev, _ = multi_ppo.train(
            environment=train_env, eval_env=eval_env, num_teachers=K,
            num_timesteps=tc.num_timesteps, episode_length=tc.episode_length,
            num_envs=K * E, num_eval_envs=K * E_ev, network_factory=network_factory,
            learning_rate=tc.learning_rate, entropy_cost=tc.entropy_cost,
            discounting=tc.discounting, unroll_length=tc.unroll_length,
            batch_size=tc.batch_size, num_minibatches=tc.num_minibatches,
            num_updates_per_batch=tc.num_updates_per_batch,
            normalize_observations=tc.normalize_observations,
            reward_scaling=tc.reward_scaling, max_grad_norm=tc.max_grad_norm,
            desired_kl=tc.desired_kl, learning_rate_schedule=tc.learning_rate_schedule,
            num_evals=tc.num_evals, seed=tc.seed + c, progress_fn=progress,
        )
        params = jax.tree_util.tree_map(lambda x: np.asarray(x)[:n_real], params)
        ev = {k: v[:n_real] for k, v in ev.items()}
        model.save_params(str(out_dir / f'chunk_{c:03d}.pkl'), {
            'teacher_ids': ids,
            'config_rows': sp.select(table, ids),
            'params': params,
            'eval': ev,
        })
        for j, tid in enumerate(ids):
            ok = (ev['return'][j] >= cfg.accept_min_return
                  and ev['terminated'][j] <= cfg.accept_max_terminated)
            rows.append(dict(teacher_id=int(tid), chunk=c, slot=j, accepted=int(ok),
                             **{k: float(v[j]) for k, v in ev.items()}))
        dt = time.time() - t_chunk
        eta = dt * (n_chunks - c - 1)
        print(f'[multi] chunk {c + 1}/{n_chunks} done in {dt / 60:.1f} min, '
              f'accepted {sum(r["accepted"] for r in rows if r["chunk"] == c)}/{n_real}, '
              f'ETA {eta / 3600:.2f} h', flush=True)

    rows.sort(key=lambda r: -r['return'])
    with open(out_dir / 'teachers_index.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(r['accepted'] for r in rows)
    summary = {
        'num_teachers': len(rows), 'accepted': n_ok,
        'return_mean': float(np.mean([r['return'] for r in rows])),
        'wall_hours': (time.time() - t_start) / 3600,
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'[multi] accepted {n_ok}/{len(rows)} teachers. Output: {out_dir}')
    logger.finish()


if __name__ == '__main__':
    main()
