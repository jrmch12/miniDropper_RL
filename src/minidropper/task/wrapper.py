"""Per-env vehicle-config wrappers.

`ConfigVmapWrapper` is a copy of mujoco_playground's
`BraxDomainRandomizationVmapWrapper` that swaps, per env, both the MJX model
(mass / inertia) *and* the non-MuJoCo `RocketParams` pytree. Env j of the batch
uses row j of `params_v` for its whole lifetime (RAPTOR: fixed dynamics per
teacher). Build `params_v` teacher-major (teacher k owns envs k*E .. k*E+E-1).
"""

import contextlib

import jax
import jax.numpy as jp
from brax.envs.wrappers import training as brax_training
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.wrapper import Wrapper

from minidropper.task.dynamics import RocketParams


def batched_model(mjx_model: mjx.Model, params_v: RocketParams, body_id: int):
    n = params_v.mass.shape[0]
    body_mass = jp.tile(mjx_model.body_mass, (n, 1)).at[:, body_id].set(params_v.mass)
    body_inertia = jp.tile(mjx_model.body_inertia, (n, 1, 1)).at[:, body_id].set(params_v.inertia)
    in_axes = jax.tree_util.tree_map(lambda x: None, mjx_model)
    in_axes = in_axes.tree_replace({'body_mass': 0, 'body_inertia': 0})
    model_v = mjx_model.tree_replace({'body_mass': body_mass, 'body_inertia': body_inertia})
    return model_v, in_axes


class ConfigVmapWrapper(Wrapper):
    """Vmaps the env over a batch of (mjx_model, RocketParams) configs."""

    def __init__(self, env: mjx_env.MjxEnv, params_v: RocketParams):
        super().__init__(env)
        self._params_v = params_v
        self._model_v, self._in_axes = batched_model(env.mjx_model, params_v, env.unwrapped._body_id)
        self.num_configs = int(params_v.mass.shape[0])

    @contextlib.contextmanager
    def v_env_fn(self, mjx_model: mjx.Model, params: RocketParams):
        env = self.env.unwrapped
        old_model, old_params = env._mjx_model, env._params
        try:
            env._mjx_model = mjx_model
            env._params = params
            yield env
        finally:
            env._mjx_model = old_model
            env._params = old_params

    def reset(self, rng: jax.Array) -> mjx_env.State:
        def reset(mjx_model, params, rng):
            with self.v_env_fn(mjx_model, params) as v_env:
                return v_env.reset(rng)

        return jax.vmap(reset, in_axes=[self._in_axes, 0, 0])(self._model_v, self._params_v, rng)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        def step(mjx_model, params, s, a):
            with self.v_env_fn(mjx_model, params) as v_env:
                return v_env.step(s, a)

        return jax.vmap(step, in_axes=[self._in_axes, 0, 0, 0])(
            self._model_v, self._params_v, state, action)


class BraxAutoResetWrapper(Wrapper):
    """Copy of mujoco_playground's BraxAutoResetWrapper (full_reset path fixed).

    With full_reset=True the playground version replaces the whole `info` on
    done, which also wipes `truncation` (PPO then treats every time-out as a
    termination) and the EpisodeWrapper statistics. Here those keys are kept
    from the finished step.
    """

    _KEEP = ('steps', 'truncation', 'episode_done', 'episode_metrics')

    def __init__(self, env, full_reset: bool = False):
        super().__init__(env)
        self._full_reset = full_reset
        self._info_key = 'AutoResetWrapper'

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng_key = jax.vmap(jax.random.split)(rng)
        rng, key = rng_key[..., 0], rng_key[..., 1]
        state = self.env.reset(key)
        state.info[f'{self._info_key}_first_data'] = state.data
        state.info[f'{self._info_key}_first_obs'] = state.obs
        state.info[f'{self._info_key}_rng'] = rng
        state.info[f'{self._info_key}_done_count'] = jp.zeros(key.shape[:-1], dtype=int)
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        rng_key = jax.vmap(jax.random.split)(state.info[f'{self._info_key}_rng'])
        reset_rng, reset_key = rng_key[..., 0], rng_key[..., 1]
        reset_state = None
        if self._full_reset:
            reset_state = self.reset(reset_key)
            reset_data, reset_obs = reset_state.data, reset_state.obs
        else:
            reset_data = state.info[f'{self._info_key}_first_data']
            reset_obs = state.info[f'{self._info_key}_first_obs']

        if 'steps' in state.info:
            steps = state.info['steps']
            state.info.update(steps=jp.where(state.done, jp.zeros_like(steps), steps))

        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape and done.shape[0] != x.shape[0]:
                return y
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jp.where(done, x, y)

        data = jax.tree.map(where_done, reset_data, state.data)
        obs = jax.tree.map(where_done, reset_obs, state.obs)

        next_info = state.info
        done_count_key = f'{self._info_key}_done_count'
        if self._full_reset and reset_state:
            next_info = jax.tree.map(where_done, reset_state.info, state.info)
            next_info[done_count_key] = state.info[done_count_key]
            for k in self._KEEP:
                if k in state.info:
                    next_info[k] = state.info[k]

        next_info[done_count_key] += state.done.astype(int)
        next_info[f'{self._info_key}_rng'] = reset_rng
        return state.replace(data=data, obs=obs, info=next_info)


def wrap_for_training(env, params_v: RocketParams, episode_length: int,
                      action_repeat: int = 1, full_reset: bool = True):
    """Same stack as mujoco_playground.wrapper.wrap_for_brax_training.

    full_reset=True so commands, actuator states and buffers in `info` are
    re-sampled on auto-reset (the default cached reset would keep them).
    """
    env = ConfigVmapWrapper(env, params_v)
    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
    env = BraxAutoResetWrapper(env, full_reset=full_reset)
    return env


def wrap_for_eval(env, params_v: RocketParams):
    """Batched env without auto-reset (episodes are masked by the caller)."""
    return ConfigVmapWrapper(env, params_v)
