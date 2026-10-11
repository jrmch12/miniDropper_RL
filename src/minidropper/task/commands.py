"""Velocity + absolute-yaw command generator.

The command target is piecewise constant (hold time U[hold_min, hold_max]) and
the reference fed to the policy / reward is a rate-limited version of it, so
steps in the target stay physically feasible.
"""

import jax
import jax.numpy as jp

from minidropper.utils.math_utils import wrap_angle


def _sample_target(cfg, key, yaw_current):
    k_mode, k_vxy, k_vz, k_yawc, k_yaw, k_hold = jax.random.split(key, 6)
    hover = jax.random.uniform(k_mode) < cfg.cmd_p_hover
    vxy = jax.random.uniform(k_vxy, (2,), minval=-cfg.cmd_vxy_max, maxval=cfg.cmd_vxy_max)
    vz = jax.random.uniform(k_vz, minval=-cfg.cmd_vz_max, maxval=cfg.cmd_vz_max)
    v = jp.where(hover, jp.zeros(3), jp.array([vxy[0], vxy[1], vz]))

    change_yaw = jax.random.uniform(k_yawc) < cfg.cmd_p_yaw_change
    yaw_new = jax.random.uniform(k_yaw, minval=-cfg.cmd_yaw_max, maxval=cfg.cmd_yaw_max)
    yaw = jp.where(change_yaw, yaw_new, yaw_current)

    hold = jax.random.uniform(k_hold, minval=cfg.cmd_hold_min, maxval=cfg.cmd_hold_max)
    return v, yaw, hold


def init_command(cfg, key, yaw0):
    k1, k2 = jax.random.split(key)
    v, yaw, hold = _sample_target(cfg, k1, yaw0)
    # Start the reference at the target so the first second is not a pure ramp.
    start_at_target = jax.random.uniform(k2) < 0.5
    return {
        'v_target': v,
        'yaw_target': yaw,
        'v_ref': jp.where(start_at_target, v, jp.zeros(3)),
        'yaw_ref': jp.where(start_at_target, yaw, yaw0),
        'timer': hold,
    }


def step_command(cfg, cmd, key, dt):
    timer = cmd['timer'] - dt
    v_new, yaw_new, hold = _sample_target(cfg, key, cmd['yaw_target'])
    resample = timer <= 0.0
    v_target = jp.where(resample, v_new, cmd['v_target'])
    yaw_target = jp.where(resample, yaw_new, cmd['yaw_target'])
    timer = jp.where(resample, hold, timer)

    dv = jp.clip(v_target - cmd['v_ref'], -cfg.cmd_accel_max * dt, cfg.cmd_accel_max * dt)
    dyaw = jp.clip(wrap_angle(yaw_target - cmd['yaw_ref']),
                   -cfg.cmd_yaw_rate_max * dt, cfg.cmd_yaw_rate_max * dt)
    return {
        'v_target': v_target,
        'yaw_target': yaw_target,
        'v_ref': cmd['v_ref'] + dv,
        'yaw_ref': wrap_angle(cmd['yaw_ref'] + dyaw),
        'timer': timer,
    }
