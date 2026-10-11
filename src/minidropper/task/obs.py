"""Observation construction.

student_obs (27, deployable, state-estimator only):
    v_world (3) | v_world - v_ref (3) | R body->world (9) | omega_body (3)
    | specific force body / g (3) | sin, cos(yaw - yaw_ref) (2) | previous action (4)
teacher_obs (31) = student_obs + EDF speeds (2) + servo angles / servo_max (2)
student_hist = last `obs_history` student_obs stacked (for the MLP+history DR baseline)
"""

import jax.numpy as jp

from minidropper.task.dynamics import GRAVITY
from minidropper.utils.math_utils import quat_to_rotmat, yaw_from_rotmat

STUDENT_OBS_DIM = 27


def kinematics(data):
    q = data.qpos[3:7]
    R = quat_to_rotmat(q / jp.linalg.norm(q))
    v_world = data.qvel[0:3]
    omega_body = data.qvel[3:6]
    return R, v_world, omega_body


def student_obs(R, v_world, omega_body, specific_force, cmd, prev_action):
    yaw_err = yaw_from_rotmat(R) - cmd['yaw_ref']
    return jp.concatenate([
        v_world,
        v_world - cmd['v_ref'],
        R.reshape(-1),
        omega_body,
        specific_force / GRAVITY,
        jp.array([jp.sin(yaw_err), jp.cos(yaw_err)]),
        prev_action,
    ])


def build_obs(params, R, v_world, omega_body, specific_force, info):
    s_obs = student_obs(R, v_world, omega_body, specific_force, info['cmd'], info['prev_action'])
    act = info['act']
    t_obs = jp.concatenate([
        s_obs,
        act['w'],
        act['servo'] / params.servo_max,
    ])
    hist = jp.concatenate([s_obs[None], info['obs_hist'][:-1]], axis=0)
    obs = {
        'policy_obs': t_obs,
        'value_obs': t_obs,
        'student_obs': s_obs,
    }
    if hist.shape[0] > 1:
        obs['student_hist'] = hist.reshape(-1)
    return obs, hist
