"""RAPTOR-style correlated sampling of miniDropper vehicle configs.

Port of rl-tools `l2f/operations_generic/10_sample_initial_parameters.h`
(sampling order: thrust-to-weight -> mass (uniform in size, m ~ size^3) ->
geometry scaled with size + reciprocal-normal noise -> inertia -> actuator
dynamics -> disturbances), adapted to a TVC rocket with two coaxial EDFs.

Each sampled config is fixed for the whole life of its teacher.

CLI:
    python -m minidropper.dr.sample_params --n 1000 --n_test 100 --seed 0 \
        --out data/params/v1.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np

G = 9.81

NOMINAL = dict(
    mass=0.40, radius=0.03, height=0.20, arm_len=0.09, com_lat=0.004,
    tw=2.0, thrust_quad=0.8, asym=0.95, tau_rise=0.05, tau_fall=0.10,
    kq=0.02, yaw_bias=0.0, servo_max=np.deg2rad(15.0), servo_tau=0.02,
    servo_rate=np.deg2rad(500.0), eta_p=2.0, act_delay=1,
    drag_lin=0.02, drag_ang=1e-5, tether_mag=0.1, tether_torsion=0.003,
    gust_std=0.0, gust_tau=0.5,
)

RANGES = dict(
    mass=(0.20, 0.70),                  # kg, uniform in cbrt(mass)
    size_deviation=0.10,                # reciprocal-normal size noise (RAPTOR mass_size_deviation)
    inertia_factor_xx=(0.7, 1.4),       # x solid-cylinder Ixx
    inertia_factor_zz=(0.7, 1.5),       # x solid-cylinder Izz
    arm_frac=(0.30, 0.70),              # pivot-to-CoM distance / tube height
    com_lat=(0.0, 0.012),               # m, lateral CoM offset (asymmetric servo layout)
    tw=(1.3, 3.0),                      # thrust-to-weight
    thrust_quad=(0.5, 1.0),
    asym=(0.8, 1.1),                    # Tmax_bottom / Tmax_top
    kq=(0.005, 0.05),                   # m
    yaw_bias_frac=(-0.05, 0.05),        # fraction of max yaw torque
    tau_rise=(0.02, 0.10),
    tau_fall=(0.03, 0.25),
    servo_max_deg=(10.0, 20.0),
    servo_tau=(0.01, 0.05),
    servo_rate_deg=(250.0, 750.0),
    servo_trim_deg=(-2.0, 2.0),
    eta_p=(1.0, 3.0),
    act_delay=(0, 2),                   # inclusive, control steps
    drag_lin=(0.0, 0.05),
    drag_ang=(0.0, 1e-4),
    tether_mag=(0.0, 0.3),              # N
    tether_torsion=(0.0, 0.01),         # Nm/rad
    gust_frac=(0.0, 1.0),               # RAPTOR: std = U * 0.3 * (tw - 1) * m g / 3
    gust_tau=(0.2, 1.0),
)


def _cyl_inertia(m, r, h):
    ixx = m * (3 * r * r + h * h) / 12.0
    izz = m * r * r / 2.0
    return np.array([ixx, ixx, izz])


def _config(mass, radius, height, inertia, arm_len, com_lat_vec, tw, thrust_quad, asym,
            tau_rise, tau_fall, kq, yaw_bias_frac, servo_max, servo_tau, servo_rate,
            servo_trim, eta_p, act_delay, drag_lin, drag_ang, tether_force,
            tether_torsion, gust_std, gust_tau):
    t_total = tw * mass * G
    t_top = t_total / (1.0 + asym)
    tmax = np.array([t_top, asym * t_top])
    yaw_bias = yaw_bias_frac * kq * t_total
    arm = np.array([-com_lat_vec[0], -com_lat_vec[1], -arm_len])
    return dict(
        mass=mass, inertia=inertia, arm=arm, tmax=tmax, thrust_quad=thrust_quad,
        tau_rise=tau_rise, tau_fall=tau_fall, kq=kq, yaw_bias=yaw_bias,
        servo_max=servo_max, servo_tau=servo_tau, servo_rate=servo_rate,
        servo_trim=np.asarray(servo_trim, dtype=np.float64), eta_p=eta_p,
        act_delay=int(act_delay), drag_lin=drag_lin, drag_ang=drag_ang,
        tether_force=np.asarray(tether_force, dtype=np.float64),
        tether_torsion=tether_torsion, gust_std=gust_std, gust_tau=gust_tau,
        # bookkeeping (not used by the env)
        radius=radius, height=height, tw=tw,
    )


def nominal_params():
    n = NOMINAL
    inertia = _cyl_inertia(n['mass'], n['radius'], n['height'])
    return _config(
        n['mass'], n['radius'], n['height'], inertia, n['arm_len'],
        np.array([n['com_lat'], 0.0]), n['tw'], n['thrust_quad'], n['asym'],
        n['tau_rise'], n['tau_fall'], n['kq'], n['yaw_bias'], n['servo_max'],
        n['servo_tau'], n['servo_rate'], np.zeros(2), n['eta_p'], n['act_delay'],
        n['drag_lin'], n['drag_ang'], np.array([0.0, 0.0, -n['tether_mag']]),
        n['tether_torsion'], n['gust_std'], n['gust_tau'])


def _reciprocal_normal(rng, dev):
    f = np.clip(rng.normal(0.0, dev), -3 * dev, 3 * dev)
    return 1.0 / (1.0 - f) if f < 0 else 1.0 + f


def sample_one(rng, R=RANGES):
    u = lambda key: rng.uniform(*R[key])
    tw = u('tw')

    size = rng.uniform(np.cbrt(R['mass'][0]), np.cbrt(R['mass'][1]))
    mass = size ** 3
    size_factor = np.cbrt(mass / NOMINAL['mass']) * _reciprocal_normal(rng, R['size_deviation'])
    radius = NOMINAL['radius'] * size_factor
    height = NOMINAL['height'] * size_factor

    inertia = _cyl_inertia(mass, radius, height)
    fxx = u('inertia_factor_xx')
    inertia = inertia * np.array([fxx, fxx, u('inertia_factor_zz')])

    arm_len = u('arm_frac') * height
    lat = u('com_lat')
    ang = rng.uniform(-np.pi, np.pi)
    com_lat_vec = lat * np.array([np.cos(ang), np.sin(ang)])

    tether_dir = np.array([*rng.normal(size=2), -rng.uniform(0.5, 2.0)])
    tether_dir /= np.linalg.norm(tether_dir)
    gust_std = u('gust_frac') * 0.3 * (tw - 1.0) * mass * G / 3.0

    return _config(
        mass, radius, height, inertia, arm_len, com_lat_vec, tw, u('thrust_quad'), u('asym'),
        u('tau_rise'), u('tau_fall'), u('kq'), u('yaw_bias_frac'),
        np.deg2rad(u('servo_max_deg')), u('servo_tau'), np.deg2rad(u('servo_rate_deg')),
        np.deg2rad(rng.uniform(*R['servo_trim_deg'], size=2)), u('eta_p'),
        rng.integers(R['act_delay'][0], R['act_delay'][1] + 1), u('drag_lin'), u('drag_ang'),
        u('tether_mag') * tether_dir, u('tether_torsion'), gust_std, u('gust_tau'))


def feasible(c):
    """Hover must be reachable with margin and trim must leave servo authority."""
    arm_len = -c['arm'][2]
    lat = np.linalg.norm(c['arm'][:2])
    trim_angle = np.arctan2(lat, arm_len) + np.max(np.abs(c['servo_trim']))
    if trim_angle > 0.5 * c['servo_max']:
        return False
    eta = np.cos(trim_angle) ** c['eta_p']
    if c['tw'] * eta < 1.2:
        return False
    # yaw authority at hover must beat bias + worst tether twist (pi/2)
    yaw_auth = c['kq'] * c['mass'] * G
    if abs(c['yaw_bias']) + c['tether_torsion'] * np.pi / 2 > 0.5 * yaw_auth:
        return False
    return True


def sample_table(n, seed, R=RANGES):
    rng = np.random.default_rng(seed)
    configs, rejected = [], 0
    while len(configs) < n:
        c = sample_one(rng, R)
        if feasible(c):
            configs.append(c)
        else:
            rejected += 1
    table = {k: np.stack([np.asarray(c[k]) for c in configs]) for k in configs[0]}
    return table, rejected


def stack_configs(configs):
    return {k: np.stack([np.asarray(c[k]) for c in configs]) for k in configs[0]}


def select(table, idx):
    return {k: v[idx] for k, v in table.items()}


def repeat_rows(table, reps):
    """Teacher-major expansion: row k is repeated `reps` times consecutively."""
    return {k: np.repeat(v, reps, axis=0) for k, v in table.items()}


def save_table(path, train, test, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f'train/{k}': v for k, v in train.items()}
    arrays.update({f'test/{k}': v for k, v in test.items()})
    np.savez(path, **arrays)
    path.with_suffix('.json').write_text(json.dumps(meta, indent=2))


def load_table(path, split='train'):
    z = np.load(path)
    pre = f'{split}/'
    return {k[len(pre):]: z[k] for k in z.files if k.startswith(pre)}


def summarize(table):
    rows = []
    for k, v in table.items():
        v = np.asarray(v, dtype=np.float64)
        rows.append(f'  {k:<15s} min {v.min():+.4g}  mean {v.mean():+.4g}  max {v.max():+.4g}')
    return '\n'.join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--n_test', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=str, default='data/params/v1.npz')
    args = ap.parse_args()

    train, rej_tr = sample_table(args.n, args.seed)
    test, rej_te = sample_table(args.n_test, args.seed + 10_000)
    meta = dict(n=args.n, n_test=args.n_test, seed=args.seed, rejected=rej_tr + rej_te,
                ranges={k: list(v) if isinstance(v, tuple) else v for k, v in RANGES.items()},
                nominal={k: float(v) for k, v in NOMINAL.items()})
    save_table(args.out, train, test, meta)
    print(f'Saved {args.n} train + {args.n_test} test configs to {args.out} '
          f'(rejected {rej_tr + rej_te} infeasible samples)')
    print(summarize(train))


if __name__ == '__main__':
    main()
