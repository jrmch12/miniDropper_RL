"""miniDropper actuator + force model.

Everything that is randomised per vehicle config and is not part of the MuJoCo
model lives in `RocketParams`. Mass and inertia are also stored here (so the
sampler has a single source of truth) and are copied into the batched MJX model
by `minidropper.task.wrapper`.

Actuators (policy action a in [-1, 1]^4):
    a[0], a[1] : top / bottom EDF throttle command -> first-order lag (rise/fall)
    a[2], a[3] : nozzle pitch / roll servo command -> lag + rate limit + trim

Forces are applied as xfrc_applied on the rocket body (world frame, at the CoM).
"""

import jax
import jax.numpy as jp
from flax import struct

from minidropper.utils.math_utils import yaw_from_rotmat

GRAVITY = 9.81
MAX_ACT_DELAY = 2  # control steps; action buffer has MAX_ACT_DELAY + 1 slots


@struct.dataclass
class RocketParams:
    mass: jax.Array            # kg
    inertia: jax.Array         # (3,) principal inertia about the CoM, kg m^2
    arm: jax.Array             # (3,) gimbal pivot position w.r.t. CoM, body frame, m
    tmax: jax.Array            # (2,) max static thrust of [top, bottom] EDF, N
    thrust_quad: jax.Array     # share of the quadratic term in the thrust curve, [0, 1]
    tau_rise: jax.Array        # EDF time constant when spinning up, s
    tau_fall: jax.Array        # EDF time constant when spinning down, s
    kq: jax.Array              # yaw torque per N of (top - bottom) thrust, m
    yaw_bias: jax.Array        # residual swirl torque, Nm
    servo_max: jax.Array       # mechanical nozzle deflection limit per axis, rad
    servo_tau: jax.Array       # servo first-order lag, s
    servo_rate: jax.Array      # servo slew-rate limit, rad/s
    servo_trim: jax.Array      # (2,) nozzle mounting offset, rad
    eta_p: jax.Array           # deflection efficiency eta = cos(delta)^eta_p
    act_delay: jax.Array       # int32 action latency in control steps
    drag_lin: jax.Array        # linear aero drag, N / (m/s)
    drag_ang: jax.Array        # angular damping, Nm / (rad/s)
    tether_force: jax.Array    # (3,) constant tether pull, world frame, N
    tether_torsion: jax.Array  # tether twist stiffness about world z, Nm / rad
    gust_std: jax.Array        # stationary std of the OU disturbance force, N
    gust_tau: jax.Array        # OU correlation time, s


PARAM_FIELDS = tuple(RocketParams.__dataclass_fields__.keys())


def params_from_dict(d, index=None):
    """Build RocketParams from a dict of arrays (optionally selecting a row)."""
    sel = (lambda x: x) if index is None else (lambda x: x[index])
    kw = {}
    for k in PARAM_FIELDS:
        v = jp.asarray(sel(d[k]))
        kw[k] = v.astype(jp.int32) if k == 'act_delay' else v.astype(jp.float32)
    return RocketParams(**kw)


def params_to_dict(p: RocketParams):
    return {k: getattr(p, k) for k in PARAM_FIELDS}


def thrust_curve(params: RocketParams, w):
    """Static thrust of each EDF for normalised speed w in [0, 1]."""
    c = params.thrust_quad
    return params.tmax * (c * w * w + (1.0 - c) * w)


def hover_speed(params: RocketParams):
    """Normalised EDF speed at which both fans together carry the weight."""
    c = params.thrust_quad
    k = params.mass * GRAVITY / jp.sum(params.tmax)
    c_safe = jp.maximum(c, 1e-3)
    w = (-(1.0 - c) + jp.sqrt((1.0 - c) ** 2 + 4.0 * c_safe * k)) / (2.0 * c_safe)
    return jp.clip(w, 0.0, 1.0)


def init_actuator_state(params: RocketParams, w0):
    return {
        'w': jp.ones(2) * w0,
        'servo': params.servo_trim,
        'gust': jp.zeros(3),
    }


def nozzle_direction(servo):
    """Unit thrust direction in body frame for nozzle pitch (about y) and roll (about x)."""
    p, r = servo[0], servo[1]
    return jp.array([jp.sin(p) * jp.cos(r), -jp.sin(r), jp.cos(p) * jp.cos(r)])


def actuator_substep(params: RocketParams, act, u_cmd, servo_target, dt):
    """Advance EDF speeds and servo angles by one physics substep."""
    w = act['w']
    tau = jp.where(u_cmd > w, params.tau_rise, params.tau_fall)
    w = w + (u_cmd - w) * (1.0 - jp.exp(-dt / tau))

    s = act['servo']
    ds = (servo_target - s) * (1.0 - jp.exp(-dt / params.servo_tau))
    ds = jp.clip(ds, -params.servo_rate * dt, params.servo_rate * dt)
    s = jp.clip(s + ds, -params.servo_max, params.servo_max)
    return {**act, 'w': w, 'servo': s}


def gust_update(params: RocketParams, gust, dt, key):
    """Ornstein-Uhlenbeck disturbance force (world frame)."""
    a = jp.exp(-dt / params.gust_tau)
    noise = jax.random.normal(key, (3,))
    return a * gust + params.gust_std * jp.sqrt(1.0 - a * a) * noise


def wrench(params: RocketParams, act, R, v_world, omega_body):
    """World-frame [force, torque] on the CoM plus diagnostics."""
    T = thrust_curve(params, act['w'])
    d_body = nozzle_direction(act['servo'])
    cos_delta = jp.clip(d_body[2], 0.0, 1.0)
    eta = cos_delta ** params.eta_p
    thrust = eta * jp.sum(T)
    F_body = thrust * d_body

    tau_body = jp.cross(params.arm, F_body)
    tau_body = tau_body + jp.array([0.0, 0.0, params.kq * (T[0] - T[1]) + params.yaw_bias])
    tau_body = tau_body - params.drag_ang * omega_body

    yaw = yaw_from_rotmat(R)
    F_world = R @ F_body - params.drag_lin * v_world + params.tether_force + act['gust']
    tau_world = R @ tau_body + jp.array([0.0, 0.0, -params.tether_torsion * yaw])

    diag = {'thrust': thrust, 'T': T, 'eta': eta, 'F_body': F_body, 'tau_body': tau_body}
    return jp.concatenate([F_world, tau_world]), diag
