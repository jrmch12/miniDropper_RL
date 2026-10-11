"""Distil accepted teachers into one GRU student (RAPTOR post-training, DAgger).

    python src/minidropper/student/train.py "teachers_dirs=[outputs/teachers/<date>/<time>]"

Logged per epoch (metrics.jsonl / console):
    train/loss             imitation loss on the training teachers' data
    train/return, ...      stats of the collection rollout (teacher-forced or student)
    val/imitation_mse      student vs held-out teachers on their configs (student acts)
    val/return, test/...   held-out teachers' configs / held-out test split, student only
Train vs val imitation loss is the overfitting curve over vehicle configs.
"""

import json
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_name='student', config_path='./cfg', version_base=None)
def main(cfg: DictConfig):
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.WARNING)
    import jax
    import jax.numpy as jnp
    import optax
    from brax.io import model

    from minidropper.dr import sample_params as sp
    from minidropper.student.training import dagger
    from minidropper.student.training.teachers import (
        load_teachers, make_teacher_network, teacher_action_fn)
    from minidropper.task.dynamics import params_from_dict
    from minidropper.task.obs import STUDENT_OBS_DIM
    from minidropper.task.rocket import MiniDropperRocket
    from minidropper.task import wrapper
    from minidropper.utils.paths import resolve
    from minidropper.utils.run_logger import RunLogger

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    tc = cfg.train
    if not cfg.teachers_dirs:
        raise ValueError('set teachers_dirs=[<train_multi run dir>, ...]')

    # ------------------------------------------------------------ teachers
    T = load_teachers([resolve(d) for d in cfg.teachers_dirs])
    n_all = len(T['ids'])
    rng = np.random.default_rng(tc.seed)
    perm = rng.permutation(n_all)
    n_val = int(round(cfg.val_fraction * n_all)) if n_all > 1 else 0
    val_idx, tr_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    take = lambda idx: (jax.tree_util.tree_map(lambda x: jnp.asarray(x[idx]), T['params']),
                        sp.select(T['config_rows'], idx))
    tparams_tr, rows_tr = take(tr_idx)
    tparams_val, rows_val = take(val_idx) if n_val else (None, None)
    n_tr = len(tr_idx)
    print(f'[student] {n_all} accepted teachers: {n_tr} train / {n_val} val')

    # ---------------------------------------------------------------- envs
    env = MiniDropperRocket(cfg=cfg.env)
    E, E_val = tc.envs_per_teacher, tc.val_envs_per_teacher
    env_tr = wrapper.wrap_for_eval(env, params_from_dict(sp.repeat_rows(rows_tr, E)))
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[1:],
                                       jax.jit(env_tr.reset)(jax.random.split(jax.random.PRNGKey(0), n_tr * E)).obs)
    tnet = make_teacher_network(T['meta'], obs_shape)

    student = dagger.make_student(cfg.student)
    rollout_tr = dagger.make_rollout(env_tr, student, teacher_action_fn(tnet, n_tr), n_tr * E, tc.episode_length)
    if n_val:
        env_val = wrapper.wrap_for_eval(env, params_from_dict(sp.repeat_rows(rows_val, E_val)))
        rollout_val = dagger.make_rollout(env_val, student, teacher_action_fn(tnet, n_val),
                                          n_val * E_val, tc.episode_length)
    test_rows = sp.load_table(resolve(cfg.params_table), 'test')
    n_test = cfg.num_test_envs
    env_test = wrapper.wrap_for_eval(env, params_from_dict(
        sp.select(test_rows, np.arange(n_test) % len(test_rows['mass']))))
    rollout_test = dagger.make_rollout(env_test, student, None, n_test, tc.episode_length)

    # ------------------------------------------------------------- student
    key = jax.random.PRNGKey(tc.seed)
    key, k_init = jax.random.split(key)
    sparams = dagger.init_student(student, k_init, STUDENT_OBS_DIM)
    optimizer = optax.chain(optax.clip_by_global_norm(tc.max_grad_norm), optax.adam(tc.learning_rate))
    opt_state = optimizer.init(sparams)
    train_epoch = dagger.make_train_epoch(student, optimizer, tc.minibatch_size)
    norm = {'mean': jnp.zeros(STUDENT_OBS_DIM), 'std': jnp.ones(STUDENT_OBS_DIM)}

    logger = RunLogger(out_dir, OmegaConf.to_container(cfg, resolve=True), tc.use_wandb,
                       tc.wandb_project, tc.wandb_run_name)
    summ = lambda d, pre: {f'{pre}/{k}': float(np.mean(np.asarray(v))) for k, v in d.items()}

    def save(name):
        model.save_params(str(out_dir / name), {
            'kind': 'gru_student', 'params': jax.tree_util.tree_map(np.asarray, sparams),
            'normalizer': {k: np.asarray(v) for k, v in norm.items()},
            'student_cfg': OmegaConf.to_container(cfg.student, resolve=True),
            'env_cfg': OmegaConf.to_container(cfg.env, resolve=True),
        })

    best = -np.inf
    for epoch in range(tc.num_epochs):
        beta = dagger.beta_schedule(epoch, tc)
        key, k_roll, k_train = jax.random.split(key, 3)
        dataset, stats = rollout_tr(sparams, norm, tparams_tr, beta, k_roll)
        if epoch == 0:
            norm = dagger.compute_normalizer(dataset)
        sparams, opt_state, loss = train_epoch(sparams, opt_state, norm, dataset, k_train)
        m = {'epoch': epoch, 'beta': beta, 'train/loss': float(loss), **summ(stats, 'train')}

        if epoch % tc.eval_every == 0 or epoch == tc.num_epochs - 1:
            key, kv, kt = jax.random.split(key, 3)
            if n_val:
                _, sv = rollout_val(sparams, norm, tparams_val, 0.0, kv)
                m.update(summ(sv, 'val'))
            _, st = rollout_test(sparams, norm, None, 0.0, kt)
            m.update(summ(st, 'test'))
            score = m.get('val/return', m['test/return'])
            if epoch >= tc.teacher_forcing_epochs and score > best:
                best = score
                save('student_best.pkl')
                m['best'] = 1

        logger.log(epoch, m, ['beta', 'train/loss', 'train/return', 'train/terminated',
                              'val/imitation_mse', 'val/return', 'val/terminated',
                              'test/return', 'test/terminated', 'test/vel_err', 'test/yaw_err'],
                   prefix='[dagger] ')

    save('student_final.pkl')
    (out_dir / 'student_meta.json').write_text(json.dumps({
        'teachers_dirs': list(cfg.teachers_dirs), 'n_train_teachers': int(n_tr),
        'n_val_teachers': int(n_val), 'val_teacher_ids': T['ids'][val_idx].tolist(),
        'best_score': float(best)}, indent=2))
    print(f'[student] saved {out_dir}/student_best.pkl and student_final.pkl')
    logger.finish()


if __name__ == '__main__':
    main()
