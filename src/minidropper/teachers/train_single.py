"""Single-policy PPO with stock brax PPO (nominal teacher, one-config teacher, DR baseline).

    python src/minidropper/teachers/train_single.py mode=nominal
    python src/minidropper/teachers/train_single.py mode=dr train.policy_obs_key=student_obs
    python src/minidropper/teachers/train_single.py mode=dr train.policy_obs_key=student_hist
"""

import functools
import json
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig, OmegaConf


def build_param_sets(cfg):
    """Returns (train_params_v, eval_params_v, description)."""
    from minidropper.dr import sample_params as sp
    from minidropper.task.dynamics import params_from_dict
    from minidropper.utils.paths import resolve

    n_tr, n_ev = cfg.train.num_envs, cfg.train.num_eval_envs
    if cfg.mode == 'nominal':
        nom = sp.nominal_params()
        tr = {k: np.repeat(np.asarray(v)[None], n_tr, axis=0) for k, v in nom.items()}
        ev = {k: np.repeat(np.asarray(v)[None], n_ev, axis=0) for k, v in nom.items()}
        desc = 'nominal config'
    elif cfg.mode == 'index':
        table = sp.load_table(resolve(cfg.params_table), 'train')
        row = sp.select(table, [cfg.config_index])
        tr = sp.repeat_rows(row, n_tr)
        ev = sp.repeat_rows(row, n_ev)
        desc = f'config {cfg.config_index} of {cfg.params_table}'
    elif cfg.mode == 'dr':
        tr, _ = sp.sample_table(n_tr, cfg.dr_seed)
        test = sp.load_table(resolve(cfg.params_table), 'test')
        ev = sp.select(test, np.arange(n_ev) % len(test['mass']))
        desc = f'DR: {n_tr} sampled configs, eval on held-out test split'
    else:
        raise ValueError(cfg.mode)
    return params_from_dict(tr), params_from_dict(ev), desc


@hydra.main(config_name='single', config_path='./cfg', version_base=None)
def main(cfg: DictConfig):
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.WARNING)
    from brax.io import model
    from minidropper.teachers.training import ppo_train as ppo

    from minidropper.task.rocket import MiniDropperRocket
    from minidropper.task import wrapper
    from minidropper.teachers.training import networks as ppo_networks
    from minidropper.utils.run_logger import RunLogger

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    tc = cfg.train

    env = MiniDropperRocket(cfg=cfg.env)
    params_tr, params_ev, desc = build_param_sets(cfg)
    print(f'[single] {desc}')
    train_env = wrapper.wrap_for_training(env, params_tr, tc.episode_length, tc.action_repeat)
    eval_env = wrapper.wrap_for_training(env, params_ev, tc.episode_length, tc.action_repeat)

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_dec_hidden_layers=tc.policy_dec_hidden_layers,
        action_hidden_layer_sizes=tc.action_hidden_layers,
        value_hidden_layer_sizes=tc.value_hidden_layers,
        activation=tc.activation,
        policy_obs_key=tc.policy_obs_key,
        value_obs_key=tc.value_obs_key,
    )

    restore_params = model.load_params(tc.restore_params_path) if tc.restore_params_path else None

    logger = RunLogger(out_dir, OmegaConf.to_container(cfg, resolve=True), tc.use_wandb,
                       tc.wandb_project, tc.wandb_run_name)
    ep_len = tc.episode_length
    print_eval = ['eval/episode_reward', 'eval/avg_episode_length',
                  'eval/track_vel_err', 'eval/track_yaw_err']
    print_train = ['episode/sum_reward', 'episode/length']

    def progress(num_steps, metrics):
        m = dict(metrics)
        # per-step averages of the tracking diagnostics (episode sums / length)
        for k in list(m):
            if k.startswith('eval/episode_track/'):
                m['eval/track_' + k.split('/')[-1]] = m[k] / np.maximum(m['eval/avg_episode_length'], 1)
        is_eval = 'eval/episode_reward' in m
        logger.log(num_steps, m, print_eval if is_eval else print_train,
                   prefix='[eval ] ' if is_eval else '[train] ')

    make_inference_fn, params, _ = ppo.train(
        environment=train_env,
        eval_env=eval_env,
        wrap_env=False,
        num_timesteps=tc.num_timesteps, num_evals=tc.num_evals, num_eval_envs=tc.num_eval_envs,
        reward_scaling=tc.reward_scaling, episode_length=ep_len,
        normalize_observations=tc.normalize_observations, action_repeat=tc.action_repeat,
        unroll_length=tc.unroll_length, num_minibatches=tc.num_minibatches,
        num_updates_per_batch=tc.num_updates_per_batch, discounting=tc.discounting,
        learning_rate=tc.learning_rate, entropy_cost=tc.entropy_cost, num_envs=tc.num_envs,
        batch_size=tc.batch_size, seed=tc.seed, log_training_metrics=tc.log_training_metrics,
        training_metrics_steps=tc.num_envs * tc.unroll_length * 50,
        learning_rate_schedule=tc.learning_rate_schedule, desired_kl=tc.desired_kl,
        max_grad_norm=tc.max_grad_norm, network_factory=network_factory,
        restore_params=restore_params, restore_value_fn=tc.restore_value_fn,
        use_pmap_on_reset=False, progress_fn=progress,
    )

    model.save_params(str(out_dir / 'trained_model.pkl'), params)
    (out_dir / 'policy_meta.json').write_text(json.dumps({
        'kind': 'ppo', 'mode': cfg.mode, 'config_index': cfg.config_index,
        'network': OmegaConf.to_container(tc, resolve=True),
        'env': OmegaConf.to_container(cfg.env, resolve=True),
    }, indent=2))
    print(f'Saved {out_dir / "trained_model.pkl"}')
    logger.finish()


if __name__ == '__main__':
    main()
