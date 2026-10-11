"""miniDropper rocket environment (MuJoCo Playground MjxEnv, MJX physics)."""

from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from ml_collections import config_dict
from omegaconf import DictConfig, OmegaConf

from mujoco_playground._src import mjx_env

from minidropper.task import commands
from minidropper.task import dynamics
from minidropper.task.obs import build_obs, kinematics, STUDENT_OBS_DIM
from minidropper.task.reward import compute_reward
from minidropper.utils.math_utils import yaw_from_rotmat

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_XML = REPO_ROOT / 'assets' / 'minidropper' / 'scene.xml'

REWARD_TERMS = ('vel', 'yaw', 'tilt', 'angvel', 'd_action', 'action')
DIAG_TERMS = ('vel_err', 'yaw_err', 'tilt', 'angvel')


def default_config() -> config_dict.ConfigDict:
    return config_dict.ConfigDict(dict(
        xml_path=str(DEFAULT_XML),
        impl='jax',
        sim_dt=0.002,
        ctrl_dt=0.01,
        obs_history=1,     # >1 adds obs['student_hist'] (MLP+history DR baseline)
        # initial state
        init_height=3.0,
        init_p_guidance=0.1,
        init_tilt_max=0.35,
        init_yaw_max=1.5,
        init_vel_max=0.5,
        init_angvel_max=1.0,
        # commands
        cmd_p_hover=0.3,
        cmd_vxy_max=1.0,
        cmd_vz_max=0.5,
        cmd_yaw_max=1.5,
        cmd_p_yaw_change=0.5,
        cmd_hold_min=1.0,
        cmd_hold_max=2.5,
        cmd_accel_max=1.5,
        cmd_yaw_rate_max=1.5,
        # reward
        reward_constant=1.5,
        termination_penalty=-100.0,
        w_vel=1.0,
        w_yaw=0.5,
        w_tilt=0.1,
        w_angvel=0.02,
        w_d_action=0.5,
        w_action=0.01,
        # termination
        term_tilt=1.2,
        term_angvel=25.0,
        term_vel_err=3.0,
    ))


def to_config_dict(cfg) -> config_dict.ConfigDict:
    base = default_config()
    if cfg is None:
        return base
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(cfg, config_dict.ConfigDict):
        cfg = cfg.to_dict()
    for k, v in cfg.items():
        if v is None and k == 'xml_path':
            continue
        base[k] = v
    return base


class MiniDropperRocket(mjx_env.MjxEnv):
    """Velocity + yaw tracking for a tethered TVC mini rocket."""

    def __init__(self, cfg=None, params: dynamics.RocketParams = None, config_overrides=None):
        cfg = to_config_dict(cfg)
        super().__init__(cfg, config_overrides=config_overrides)
        self.cfg = self._config

        self._xml_path = str(self.cfg.xml_path)
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self.cfg.impl)
        self._body_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, 'rocket')

        if params is None:
            from minidropper.dr.sample_params import nominal_params
            params = dynamics.params_from_dict(nominal_params())
        self._params = params

    # ------------------------------------------------------------------ reset
    def reset(self, rng: jax.Array) -> mjx_env.State:
        cfg, p = self.cfg, self._params
        rng, k_g, k_tilt, k_axis, k_yaw, k_v, k_w, k_cmd, k_hover = jax.random.split(rng, 9)

        guided = jax.random.uniform(k_g) < cfg.init_p_guidance
        tilt = jax.random.uniform(k_tilt, minval=0.0, maxval=cfg.init_tilt_max)
        axis_ang = jax.random.uniform(k_axis, minval=-jp.pi, maxval=jp.pi)
        yaw = jax.random.uniform(k_yaw, minval=-cfg.init_yaw_max, maxval=cfg.init_yaw_max)
        tilt = jp.where(guided, 0.0, tilt)

        # yaw about world z, then tilt about a horizontal axis
        q_yaw = jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])
        ax = jp.array([jp.cos(axis_ang), jp.sin(axis_ang), 0.0])
        q_tilt = jp.concatenate([jp.array([jp.cos(tilt / 2)]), jp.sin(tilt / 2) * ax])
        quat = _quat_mul(q_tilt, q_yaw)

        kv1, kv2 = jax.random.split(k_v)
        v0 = jax.random.uniform(kv1, (3,), minval=-cfg.init_vel_max, maxval=cfg.init_vel_max)
        w0 = jax.random.uniform(kv2, (3,), minval=-cfg.init_angvel_max, maxval=cfg.init_angvel_max)
        v0 = jp.where(guided, 0.0, v0)
        w0 = jp.where(guided, 0.0, w0)

        qpos = jp.concatenate([jp.array([0.0, 0.0, cfg.init_height]), quat])
        qvel = jp.concatenate([v0, w0])
        data = mjx_env.make_data(self._mj_model, qpos=qpos, qvel=qvel, impl=self._mjx_model.impl.value)
        data = mjx.forward(self._mjx_model, data)

        w_hover = dynamics.hover_speed(p)
        w_init = jp.clip(w_hover * jax.random.uniform(k_w, (), minval=0.9, maxval=1.1), 0.0, 1.0)
        act = dynamics.init_actuator_state(p, w_init)
        a_hover = jp.array([2 * w_hover - 1, 2 * w_hover - 1, 0.0, 0.0])

        R, v_world, omega_body = kinematics(data)
        cmd = commands.init_command(cfg, k_cmd, yaw_from_rotmat(R))
        specific_force = R.T @ jp.array([0.0, 0.0, dynamics.GRAVITY])

        info = {
            'rng': rng,
            'step': jp.array(0, dtype=jp.int32),
            'act': act,
            'act_buf': jp.tile(a_hover, (dynamics.MAX_ACT_DELAY + 1, 1)),
            'prev_action': a_hover,
            'v_prev': v_world,
            'cmd': cmd,
            'obs_hist': jp.zeros((cfg.obs_history, STUDENT_OBS_DIM)),
            'diag': self._empty_diag(),
        }
        obs, hist = build_obs(p, R, v_world, omega_body, specific_force, info)
        hist = jp.tile(obs['student_obs'], (cfg.obs_history, 1))
        info['obs_hist'] = hist
        if cfg.obs_history > 1:
            obs['student_hist'] = hist.reshape(-1)

        metrics = {f'reward/{k}': jp.zeros(()) for k in REWARD_TERMS}
        metrics.update({f'track/{k}': jp.zeros(()) for k in DIAG_TERMS})
        return mjx_env.State(data=data, obs=obs, reward=jp.zeros(()), done=jp.zeros(()),
                             metrics=metrics, info=info)

    # ------------------------------------------------------------------- step
    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        cfg, p = self.cfg, self._params
        info = state.info
        rng, k_gust, k_cmd = jax.random.split(info['rng'], 3)
        action = jp.clip(action, -1.0, 1.0)

        act_buf = jp.concatenate([action[None], info['act_buf'][:-1]], axis=0)
        applied = act_buf[p.act_delay]
        u_cmd = (applied[:2] + 1.0) * 0.5
        servo_target = applied[2:] * p.servo_max + p.servo_trim

        act = dict(info['act'])
        act['gust'] = dynamics.gust_update(p, act['gust'], self.dt, k_gust)

        def substep(carry, _):
            data, act = carry
            act = dynamics.actuator_substep(p, act, u_cmd, servo_target, self.sim_dt)
            R, v, om = kinematics(data)
            xfrc, diag = dynamics.wrench(p, act, R, v, om)
            data = data.replace(xfrc_applied=data.xfrc_applied.at[self._body_id].set(xfrc))
            data = mjx.step(self._mjx_model, data)
            return (data, act), diag

        (data, act), diags = jax.lax.scan(substep, (state.data, act), (), length=self.n_substeps)

        R, v_world, omega_body = kinematics(data)
        accel_world = (v_world - info['v_prev']) / self.dt
        specific_force = R.T @ (accel_world + jp.array([0.0, 0.0, dynamics.GRAVITY]))

        reward, terms, terminated, track = compute_reward(
            cfg, R, v_world, omega_body, info['cmd'], action, info['prev_action'])

        cmd = commands.step_command(cfg, info['cmd'], k_cmd, self.dt)
        new_info = {
            **info,
            'rng': rng,
            'step': info['step'] + 1,
            'act': act,
            'act_buf': act_buf,
            'prev_action': action,
            'v_prev': v_world,
            'cmd': cmd,
            'diag': {
                'thrust': diags['thrust'][-1],
                'T': diags['T'][-1],
                'eta': diags['eta'][-1],
                'servo': act['servo'],
                'w': act['w'],
                'gust': act['gust'],
                **{f'r_{k}': v for k, v in terms.items()},
                **track,
            },
        }
        obs, hist = build_obs(p, R, v_world, omega_body, specific_force, new_info)
        new_info['obs_hist'] = hist

        metrics = dict(state.metrics)
        metrics.update({f'reward/{k}': terms[k] for k in REWARD_TERMS})
        metrics.update({f'track/{k}': track[k] for k in DIAG_TERMS})
        obs = jax.tree.map(lambda x: jp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), obs)
        return state.replace(data=data, obs=obs, reward=reward,
                             done=terminated.astype(jp.float32), metrics=metrics, info=new_info)

    # ---------------------------------------------------------------- helpers
    def _empty_diag(self):
        z = jp.zeros(())
        d = {'thrust': z, 'T': jp.zeros(2), 'eta': z, 'servo': jp.zeros(2),
             'w': jp.zeros(2), 'gust': jp.zeros(3)}
        d.update({f'r_{k}': z for k in REWARD_TERMS})
        d.update({k: z for k in DIAG_TERMS})
        return d

    @property
    def params(self) -> dynamics.RocketParams:
        return self._params

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return 4

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return jp.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
