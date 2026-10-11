"""RAPTOR-style squared-cost reward:  r = C - sum_i w_i c_i,  termination -> penalty."""

import jax.numpy as jp

from minidropper.utils.math_utils import tilt_from_rotmat, wrap_angle, yaw_from_rotmat


def compute_reward(cfg, R, v_world, omega_body, cmd, action, prev_action):
    vel_err = v_world - cmd['v_ref']
    yaw_err = wrap_angle(yaw_from_rotmat(R) - cmd['yaw_ref'])
    tilt = tilt_from_rotmat(R)

    costs = {
        'vel': jp.sum(vel_err ** 2),
        'yaw': 1.0 - jp.cos(yaw_err),
        'tilt': tilt ** 2,
        'angvel': jp.sum(omega_body ** 2),
        'd_action': jp.sum((action - prev_action) ** 2),
        'action': jp.sum(action[2:] ** 2),
    }
    weights = {
        'vel': cfg.w_vel, 'yaw': cfg.w_yaw, 'tilt': cfg.w_tilt,
        'angvel': cfg.w_angvel, 'd_action': cfg.w_d_action, 'action': cfg.w_action,
    }
    terms = {k: -weights[k] * costs[k] for k in costs}

    terminated = (
        (tilt > cfg.term_tilt)
        | (jp.linalg.norm(omega_body) > cfg.term_angvel)
        | (jp.linalg.norm(vel_err) > cfg.term_vel_err)
        | ~jp.all(jp.isfinite(v_world))
    )
    r = cfg.reward_constant + sum(terms.values())
    r = jp.where(terminated, cfg.termination_penalty, r)
    r = jp.nan_to_num(r, nan=cfg.termination_penalty)

    diag = {
        'vel_err': jp.linalg.norm(vel_err),
        'yaw_err': jp.abs(yaw_err),
        'tilt': tilt,
        'angvel': jp.linalg.norm(omega_body),
    }
    return r, terms, terminated, diag
