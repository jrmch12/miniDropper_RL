"""Quaternion / rotation helpers (MuJoCo convention: q = [w, x, y, z])."""

import jax.numpy as jp


def euler_to_quaternion(roll, pitch, yaw):
    cr, sr = jp.cos(roll * 0.5), jp.sin(roll * 0.5)
    cp, sp = jp.cos(pitch * 0.5), jp.sin(pitch * 0.5)
    cy, sy = jp.cos(yaw * 0.5), jp.sin(yaw * 0.5)
    return jp.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def quat_to_rotmat(q):
    """Body-to-world rotation matrix."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def yaw_from_rotmat(R):
    """Heading of the body x-axis projected on the world xy-plane."""
    return jp.arctan2(R[1, 0], R[0, 0])


def tilt_from_rotmat(R):
    """Angle between body z-axis and world z-axis."""
    return jp.arccos(jp.clip(R[2, 2], -1.0, 1.0))


def wrap_angle(a):
    return jp.arctan2(jp.sin(a), jp.cos(a))
