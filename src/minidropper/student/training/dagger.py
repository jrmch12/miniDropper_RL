"""RAPTOR post-training: distil N teachers into one GRU student (DAgger).

Per epoch (rl-tools src/foundation_policy/post_training/main.cpp):
  1. roll out every env for one full episode from h = 0. Env j runs teacher
     j // E's vehicle config. The executed action is the teacher's with
     probability beta (beta = 1 during the teacher-forcing epochs, RAPTOR: 10),
     the student's otherwise (on-policy DAgger).
  2. every visited state is labelled with the deterministic action of that
     env's teacher; steps after a termination are masked out.
  3. one shuffled pass over the episodes: minibatches of whole sequences,
     full-episode BPTT, MSE on actions (RAPTOR: batch 64, lr 1e-4).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax

from minidropper.common.networks import GRUPolicy


def make_student(cfg):
    return GRUPolicy(hidden_dim=cfg.hidden_dim, input_dim=cfg.input_dim, action_size=4)


def init_student(student, key, obs_dim):
    h0 = jnp.zeros((1, student.hidden_dim))
    return student.init(key, h0, jnp.zeros((1, obs_dim)))


def normalize(norm, x):
    return (x - norm['mean']) / norm['std']


def student_step(student, sparams, norm, h, obs):
    h, out = student.apply(sparams, h, normalize(norm, obs))
    return h, jnp.tanh(out)


def make_rollout(env, student, teacher_act, num_envs, episode_length):
    """Returns jitted rollout(sparams, norm, tparams, beta, key) -> (dataset, stats).

    teacher_act may be None (evaluation-only env without teachers).
    """

    def rollout(sparams, norm, tparams, beta, key):
        k_reset, k_mix, k_scan = jax.random.split(key, 3)
        state = env.reset(jax.random.split(k_reset, num_envs))
        h = jnp.zeros((num_envs, student.hidden_dim))

        def step(carry, k):
            s, h, alive = carry
            s_obs = s.obs['student_obs']
            h, a_student = student_step(student, sparams, norm, h, s_obs)
            if teacher_act is not None:
                a_teacher = teacher_act(tparams, s.obs)
                use_teacher = jax.random.uniform(k, (num_envs, 1)) < beta
                a = jnp.where(use_teacher, a_teacher, a_student)
            else:
                a_teacher = a_student
                a = a_student
            ns = env.step(s, a)
            d = ns.info['diag']
            on = alive > 0
            rec = {
                'obs': s_obs,
                'target': a_teacher,
                'mask': alive,
                'reward': jnp.where(on, ns.reward, 0.0),
                'vel_err': jnp.where(on, d['vel_err'], 0.0),
                'yaw_err': jnp.where(on, d['yaw_err'], 0.0),
                'sq_err': jnp.where(on, jnp.mean((a_student - a_teacher) ** 2, axis=-1), 0.0),
            }
            alive = alive * (1.0 - ns.done)
            return (ns, h, alive), rec

        keys = jax.random.split(k_scan, episode_length)
        (_, _, alive), data = jax.lax.scan(step, (state, h, jnp.ones(num_envs)), keys)
        length = jnp.maximum(jnp.sum(data['mask'], axis=0), 1.0)          # (B,)
        per_env = {
            'return': jnp.sum(data['reward'], axis=0),
            'length': jnp.sum(data['mask'], axis=0),
            'vel_err': jnp.sum(data['vel_err'], axis=0) / length,
            'yaw_err': jnp.sum(data['yaw_err'], axis=0) / length,
            'imitation_mse': jnp.sum(data['sq_err'], axis=0) / length,
            'terminated': 1.0 - alive,
        }
        dataset = {k: data[k] for k in ('obs', 'target', 'mask')}
        return dataset, per_env

    return jax.jit(rollout)


def make_train_epoch(student, optimizer, minibatch_size):
    """One shuffled pass over the episodes of the dataset (T, B, ...)."""

    def seq_loss(sparams, norm, obs, target, mask):
        B = obs.shape[1]
        h0 = jnp.zeros((B, student.hidden_dim))

        def f(h, x):
            h, a = student_step(student, sparams, norm, h, x)
            return h, a

        _, a = jax.lax.scan(f, h0, obs)
        err = jnp.mean((a - target) ** 2, axis=-1)
        return jnp.sum(err * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    grad_fn = jax.value_and_grad(seq_loss)

    def train_epoch(sparams, opt_state, norm, dataset, key):
        B = dataset['obs'].shape[1]
        n_mb = max(B // minibatch_size, 1)
        perm = jax.random.permutation(key, B)[: n_mb * minibatch_size]

        def mb(carry, idx):
            sparams, opt_state = carry
            batch = {k: v[:, idx] for k, v in dataset.items()}
            loss, g = grad_fn(sparams, norm, batch['obs'], batch['target'], batch['mask'])
            updates, opt_state = optimizer.update(g, opt_state, sparams)
            sparams = optax.apply_updates(sparams, updates)
            return (sparams, opt_state), loss

        (sparams, opt_state), losses = jax.lax.scan(
            mb, (sparams, opt_state), perm.reshape(n_mb, minibatch_size))
        return sparams, opt_state, jnp.mean(losses)

    return jax.jit(train_epoch)


def compute_normalizer(dataset, std_min=1e-3):
    obs = np.asarray(dataset['obs'])
    m = np.asarray(dataset['mask'])[..., None]
    n = max(m.sum(), 1.0)
    mean = (obs * m).sum(axis=(0, 1)) / n
    var = (((obs - mean) ** 2) * m).sum(axis=(0, 1)) / n
    return {'mean': jnp.asarray(mean), 'std': jnp.asarray(np.maximum(np.sqrt(var), std_min))}


def beta_schedule(epoch, cfg):
    if epoch < cfg.teacher_forcing_epochs:
        return 1.0
    t = epoch - cfg.teacher_forcing_epochs
    frac = min(t / max(cfg.beta_decay_epochs, 1), 1.0)
    return float(cfg.beta_start + frac * (cfg.beta_end - cfg.beta_start))
