"""Multi-teacher PPO: K independent PPO learners trained in one jitted loop.

Derived from brax 0.14.2 brax/training/agents/ppo/train.py (see ppo_train.py
for the verbatim copy). Changes w.r.t. brax:
  * single device, jax.jit instead of pmap;
  * every TrainingState leaf has a leading teacher axis K (vmapped init);
  * the env batch is K * E envs, teacher-major: envs k*E .. k*E+E-1 all run
    teacher k's fixed vehicle config (see minidropper.task.wrapper);
  * acting: obs are reshaped to (K, E, ...) and the policy is vmapped over the
    stacked params; the physics itself is vmapped once, never twice;
  * learning: rollout data is regrouped to (K, E * n_unrolls, T, ...) and the
    brax sgd_step / minibatch_step are vmapped over K (per-teacher adaptive-KL
    learning rate and per-teacher observation normaliser);
  * eval returns per-teacher metrics (return, length, tracking errors).

Losses, GAE, gradients, acting and running statistics are imported from brax.
"""

import functools
import time
from typing import Callable, Optional

from brax.training import acting
from brax.training import gradients
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import optimizer as ppo_optimizer
from brax.training import types
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from minidropper.teachers.training import networks as ppo_networks


@flax.struct.dataclass
class TrainingState:
    optimizer_state: optax.OptState
    params: ppo_losses.PPONetworkParams
    normalizer_params: running_statistics.RunningStatisticsState


def _reshape_teacher(x, K):
    return x.reshape((K, x.shape[0] // K) + x.shape[1:])


def _flatten_teacher(x):
    return x.reshape((-1,) + x.shape[2:])


def train(
    environment,
    eval_env,
    num_teachers: int,
    num_timesteps: int,
    episode_length: int,
    num_envs: int,
    num_eval_envs: int,
    network_factory: Callable = ppo_networks.make_ppo_networks,
    learning_rate: float = 3e-4,
    entropy_cost: float = 1e-3,
    discounting: float = 0.99,
    unroll_length: int = 32,
    batch_size: int = 32,
    num_minibatches: int = 4,
    num_updates_per_batch: int = 4,
    normalize_observations: bool = True,
    reward_scaling: float = 0.1,
    clipping_epsilon: float = 0.3,
    gae_lambda: float = 0.95,
    max_grad_norm: Optional[float] = 1.0,
    normalize_advantage: bool = True,
    vf_loss_coefficient: float = 0.5,
    desired_kl: float = 0.02,
    learning_rate_schedule: Optional[str] = 'ADAPTIVE_KL',
    learning_rate_schedule_min_lr: float = 1e-5,
    learning_rate_schedule_max_lr: float = 1e-2,
    num_evals: int = 5,
    seed: int = 0,
    progress_fn: Callable = lambda *args: None,
):
    """Returns (stacked params (normalizer, policy, value), last eval dict, make_policy)."""
    K = num_teachers
    assert num_envs % K == 0 and num_eval_envs % K == 0
    E = num_envs // K
    E_eval = num_eval_envs // K
    assert (batch_size * num_minibatches) % E == 0, 'batch_size * num_minibatches must be a multiple of envs per teacher'
    n_unrolls = batch_size * num_minibatches // E

    env_steps_per_training_step = batch_size * unroll_length * num_minibatches  # per teacher
    num_evals_after_init = max(num_evals - 1, 1)
    num_training_steps_per_epoch = int(np.ceil(
        num_timesteps / (num_evals_after_init * env_steps_per_training_step)))

    key = jax.random.PRNGKey(seed)
    key_init, key_env, key_eval, local_key = jax.random.split(key, 4)

    env = environment
    env_state = jax.jit(env.reset)(jax.random.split(key_env, num_envs))

    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[1:], env_state.obs)
    normalize = running_statistics.normalize if normalize_observations else (lambda x, y: x)
    ppo_network = network_factory(obs_shape, env.action_size, preprocess_observations_fn=normalize)
    make_policy = ppo_networks.make_inference_fn(ppo_network)

    # ---------------------------------------------------------------- optimizer
    lr_is_adaptive_kl = (ppo_optimizer.LRSchedule(learning_rate_schedule or 'NONE')
                         == ppo_optimizer.LRSchedule.ADAPTIVE_KL)
    base_optimizer = (optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate)
                      if lr_is_adaptive_kl else optax.adam(learning_rate=learning_rate))
    optimizer = (optax.chain(optax.clip_by_global_norm(max_grad_norm), base_optimizer)
                 if max_grad_norm is not None else base_optimizer)

    loss_fn = functools.partial(
        ppo_losses.compute_ppo_loss,
        ppo_network=ppo_network,
        entropy_cost=entropy_cost,
        discounting=discounting,
        reward_scaling=reward_scaling,
        gae_lambda=gae_lambda,
        clipping_epsilon=clipping_epsilon,
        normalize_advantage=normalize_advantage,
        vf_coefficient=vf_loss_coefficient,
    )
    loss_and_pgrad_fn = gradients.loss_and_pgrad(loss_fn, pmap_axis_name=None, has_aux=True)

    # ------------------------------------------------------------------- init
    obs_spec = jax.tree_util.tree_map(
        lambda x: specs.Array(x.shape[-1:], jnp.dtype('float32')), env_state.obs)

    def init_one(k):
        kp, kv = jax.random.split(k)
        params = ppo_losses.PPONetworkParams(
            policy=ppo_network.policy_network.init(kp),
            value=ppo_network.value_network.init(kv),
        )
        return TrainingState(
            optimizer_state=optimizer.init(params),
            params=params,
            normalizer_params=running_statistics.init_state(obs_spec),
        )

    training_state = jax.jit(jax.vmap(init_one))(jax.random.split(key_init, K))

    # ----------------------------------------------------------------- acting
    def batched_policy(normalizer_params, policy_params, deterministic=False):
        def policy(obs, key):
            obs_k = jax.tree_util.tree_map(lambda x: _reshape_teacher(x, K), obs)
            keys = jax.random.split(key, K)

            def one(norm, pol, o, k):
                return make_policy((norm, pol), deterministic=deterministic)(o, k)

            actions, extras = jax.vmap(one)(normalizer_params, policy_params, obs_k, keys)
            return _flatten_teacher(actions), jax.tree_util.tree_map(_flatten_teacher, extras)

        return policy

    # --------------------------------------------------------------- learning
    def minibatch_step(carry, data, normalizer_params):
        optimizer_state, params, key = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), grads = loss_and_pgrad_fn(params, normalizer_params, data, key_loss)
        if lr_is_adaptive_kl:
            optimizer_state, lr = ppo_optimizer.adaptive_kl_learning_rate(
                optimizer_state, metrics['kl_mean'], desired_kl,
                min_learning_rate=learning_rate_schedule_min_lr,
                max_learning_rate=learning_rate_schedule_max_lr)
        else:
            lr = jnp.array(learning_rate)
        metrics['learning_rate'] = lr
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        params = optax.apply_updates(params, updates)
        return (optimizer_state, params, key), metrics

    def sgd_step(carry, unused_t, data, normalizer_params):
        optimizer_state, params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        def convert_data(x):
            x = jax.random.permutation(key_perm, x)
            return jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])

        shuffled = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(minibatch_step, normalizer_params=normalizer_params),
            (optimizer_state, params, key_grad), shuffled, length=num_minibatches)
        return (optimizer_state, params, key), metrics

    def update_teacher(ts: TrainingState, data: types.Transition, key):
        normalizer_params = ts.normalizer_params
        if not lr_is_adaptive_kl:
            normalizer_params = running_statistics.update(normalizer_params, data.observation)
        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(sgd_step, data=data, normalizer_params=normalizer_params),
            (ts.optimizer_state, ts.params, key), (), length=num_updates_per_batch)
        if lr_is_adaptive_kl:
            normalizer_params = running_statistics.update(normalizer_params, data.observation)

        # training-episode statistics of this teacher (episodes finished in this batch)
        done = data.extras['state_extras']['episode_done']
        ep = data.extras['state_extras']['episode_metrics']
        n_done = jnp.maximum(jnp.sum(done), 1.0)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        metrics['train_return'] = jnp.sum(ep['sum_reward'] * done) / n_done
        metrics['train_length'] = jnp.sum(ep['length'] * done) / n_done
        metrics['train_episodes'] = jnp.sum(done)
        return TrainingState(optimizer_state, params, normalizer_params), metrics

    def training_step(carry, unused_t):
        ts, state, key = carry
        key_sgd, key_unroll, new_key = jax.random.split(key, 3)
        policy = batched_policy(ts.normalizer_params, ts.params.policy)

        def f(carry, unused_t):
            s, k = carry
            k, nk = jax.random.split(k)
            s, data = acting.generate_unroll(
                env, s, policy, k, unroll_length,
                extra_fields=('truncation', 'episode_metrics', 'episode_done'))
            return (s, nk), data

        (state, _), data = jax.lax.scan(f, (state, key_unroll), (), length=n_unrolls)
        # (n_unrolls, T, K*E, ...) -> (K, n_unrolls * E, T, ...)
        def regroup(x):
            x = x.reshape((n_unrolls, unroll_length, K, E) + x.shape[3:])
            x = jnp.moveaxis(x, 2, 0)          # K, n_unrolls, T, E
            x = jnp.swapaxes(x, 2, 3)          # K, n_unrolls, E, T
            return x.reshape((K, n_unrolls * E, unroll_length) + x.shape[4:])

        data = jax.tree_util.tree_map(regroup, data)
        ts, metrics = jax.vmap(update_teacher)(ts, data, jax.random.split(key_sgd, K))
        return (ts, state, new_key), metrics

    def training_epoch(ts, state, key):
        (ts, state, _), metrics = jax.lax.scan(
            training_step, (ts, state, key), (), length=num_training_steps_per_epoch)
        # mean over training steps, keep teacher axis
        metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), metrics)
        return ts, state, metrics

    training_epoch = jax.jit(training_epoch, donate_argnums=(0, 1))

    # ------------------------------------------------------------------- eval
    @jax.jit
    def run_eval(ts: TrainingState, key):
        k_reset, k_act = jax.random.split(key)
        state = eval_env.reset(jax.random.split(k_reset, num_eval_envs))
        policy = batched_policy(ts.normalizer_params, ts.params.policy, deterministic=True)

        def step(carry, _):
            s, alive, acc = carry
            a, _ = policy(s.obs, k_act)
            s = eval_env.step(s, a)
            d = s.info['diag']
            on = alive > 0
            add = lambda x: jnp.where(on, x, 0.0)
            acc = {
                'return': acc['return'] + add(s.reward),
                'length': acc['length'] + alive,
                'vel_err': acc['vel_err'] + add(d['vel_err']),
                'yaw_err': acc['yaw_err'] + add(d['yaw_err']),
                'tilt': acc['tilt'] + add(d['tilt']),
            }
            alive = alive * (1.0 - s.done)
            return (s, alive, acc), ()

        z = jnp.zeros(num_eval_envs)
        acc0 = {k: z for k in ('return', 'length', 'vel_err', 'yaw_err', 'tilt')}
        (_, alive, acc), _ = jax.lax.scan(step, (state, jnp.ones(num_eval_envs), acc0), (),
                                          length=episode_length)
        length = jnp.maximum(acc['length'], 1.0)
        per_env = {
            'return': acc['return'],
            'length': acc['length'],
            'vel_err': acc['vel_err'] / length,
            'yaw_err': acc['yaw_err'] / length,
            'tilt': acc['tilt'] / length,
            'terminated': 1.0 - alive,
        }
        return jax.tree_util.tree_map(lambda x: jnp.mean(x.reshape(K, E_eval), axis=1), per_env)

    # ------------------------------------------------------------------- loop
    def summarize(step, train_metrics, eval_metrics, wall):
        m = {'wall': wall, 'env_steps_per_teacher': step}
        for name, v in eval_metrics.items():
            v = np.asarray(v)
            m[f'eval/{name}'] = v
            m[f'eval/{name}_mean'] = float(v.mean())
            m[f'eval/{name}_min'] = float(v.min())
            m[f'eval/{name}_max'] = float(v.max())
        for name, v in (train_metrics or {}).items():
            v = np.asarray(v)
            m[f'train/{name}'] = v if name.startswith('train_') else None
            m[f'train/{name}_mean'] = float(v.mean())
        return {k: v for k, v in m.items() if v is not None}

    t0 = time.time()
    local_key, k = jax.random.split(local_key)
    eval_metrics = run_eval(training_state, k)
    progress_fn(0, summarize(0, None, eval_metrics, time.time() - t0))

    current_step = 0
    for it in range(num_evals_after_init):
        epoch_key, local_key = jax.random.split(local_key)
        t = time.time()
        training_state, env_state, train_metrics = training_epoch(training_state, env_state, epoch_key)
        jax.block_until_ready(train_metrics)
        epoch_time = time.time() - t
        current_step += num_training_steps_per_epoch * env_steps_per_training_step
        local_key, k = jax.random.split(local_key)
        eval_metrics = run_eval(training_state, k)
        m = summarize(current_step, train_metrics, eval_metrics, time.time() - t0)
        m['sps_total'] = num_training_steps_per_epoch * env_steps_per_training_step * K / epoch_time
        progress_fn(current_step, m)

    params = (training_state.normalizer_params, training_state.params.policy,
              training_state.params.value)
    return params, jax.tree_util.tree_map(np.asarray, eval_metrics), make_policy
