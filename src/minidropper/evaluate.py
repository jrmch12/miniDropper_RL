"""Evaluate any miniDropper policy (single PPO, one multi-teacher teacher, GRU student).

GUI (interactive MuJoCo viewer, keyboard velocity / yaw commands):
    python -m minidropper.evaluate --policy <ckpt> --mode gui [--config nominal|teacher|train:3|test:5]

    Keys: arrows = vx / vy command, PageUp / PageDown = vz, ',' / '.' = yaw left / right,
          Insert = toggle manual / random commands, Backspace = reset episode, Home = hover (zero cmd)

Video (headless MP4 of one rollout, same overlay arrows as the GUI, random commands):
    python -m minidropper.evaluate --policy <ckpt> --mode video [--steps 1000] [--out flight.mp4]

Batch (many episodes in parallel, summary + optional full dump for AI-driven tuning):
    python -m minidropper.evaluate --policy <ckpt> --mode batch --configs test --episodes 512
    python -m minidropper.evaluate --policy <ckpt> --mode batch --configs test --ai --out outputs/eval/x

    --configs: nominal | teacher | train | test | train:<i> | test:<i>
    --ai     : prints per-term reward breakdown, worst episodes (with their vehicle params and the
               last steps before termination), param-vs-return correlations, and writes
               summary.json + traces.npz (every step of --trace episodes) to --out.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

REWARD_TERMS = ('vel', 'yaw', 'tilt', 'angvel', 'd_action', 'action')
SCALAR_PARAMS = ('mass', 'tw', 'thrust_quad', 'tau_rise', 'tau_fall', 'kq', 'yaw_bias',
                 'servo_max', 'servo_tau', 'servo_rate', 'eta_p', 'act_delay', 'drag_lin',
                 'tether_torsion', 'gust_std', 'height')


# ----------------------------------------------------------------------------- setup
def _config_rows(spec, n, table_path, pol):
    from minidropper.dr import sample_params as sp
    if spec == 'nominal':
        row = sp.nominal_params()
        return {k: np.repeat(np.asarray(v)[None], n, axis=0) for k, v in row.items()}
    if spec == 'teacher':
        if pol.config_row is None:
            raise ValueError('--configs teacher needs a multi-teacher checkpoint')
        return {k: np.repeat(np.asarray(v)[None], n, axis=0) for k, v in pol.config_row.items()}
    split, _, idx = spec.partition(':')
    table = sp.load_table(table_path, split)
    n_tab = len(table['mass'])
    rows = np.full(n, int(idx)) if idx else np.arange(n) % n_tab
    return sp.select(table, rows)


def _build(args, n_envs):
    import jax
    from minidropper.policies import load_policy, peek_env_cfg
    from minidropper.task.dynamics import params_from_dict
    from minidropper.task.rocket import MiniDropperRocket
    from minidropper.task import wrapper
    from minidropper.utils.paths import resolve

    env_cfg = dict(peek_env_cfg(args.policy))
    env = MiniDropperRocket(cfg=env_cfg)
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape, jax.eval_shape(env.reset, jax.random.PRNGKey(0)).obs)
    pol = load_policy(args.policy, obs_shape, args.teacher_id)
    rows = _config_rows(args.configs, n_envs, resolve(args.params_table), pol)
    venv = wrapper.wrap_for_eval(env, params_from_dict(rows))
    return env, venv, pol, rows


def _trace_step(s, a):
    d = s.info['diag']
    c = s.info['cmd']
    return {
        'qpos': s.data.qpos[:, :7], 'qvel': s.data.qvel[:, :6],
        'v_ref': c['v_ref'], 'yaw_ref': c['yaw_ref'], 'action': a,
        'w': d['w'], 'servo': d['servo'], 'thrust': d['thrust'], 'T': d['T'], 'gust': d['gust'],
        'reward': s.reward, 'done': s.done,
        'vel_err': d['vel_err'], 'yaw_err': d['yaw_err'], 'tilt': d['tilt'], 'angvel': d['angvel'],
        **{f'r_{k}': d[f'r_{k}'] for k in REWARD_TERMS},
    }


# ----------------------------------------------------------------------------- batch
def run_batch(args):
    import jax
    import jax.numpy as jnp

    n = args.episodes
    env, venv, pol, rows = _build(args, n)
    T = args.steps or 500
    print(f'[eval] {pol.description}\n[eval] {n} episodes x {T} steps on configs "{args.configs}"')

    @jax.jit
    def rollout(key):
        state = venv.reset(jax.random.split(key, n))
        carry = pol.init_carry(n)

        def step(c, _):
            s, h = c
            a, h = pol.act(s.obs, h)
            ns = venv.step(s, a)
            return (ns, h), _trace_step(ns, a)

        _, tr = jax.lax.scan(step, (state, carry), (), length=T)
        return tr

    t = time.time()
    tr = jax.tree_util.tree_map(np.asarray, rollout(jax.random.PRNGKey(args.seed)))
    print(f'[eval] rollout {time.time() - t:.1f}s (incl. compile)')

    done = tr['done'] > 0                                        # (T, n)
    first_done = np.where(done.any(0), done.argmax(0), T)        # step index of termination
    alive = np.arange(T)[:, None] <= first_done[None]            # includes the terminal step
    L = alive.sum(0).clip(1)
    sum_a = lambda x: np.where(alive, np.nan_to_num(x), 0.0).sum(0)
    mean_a = lambda x: sum_a(x) / L
    ep = {
        'return': sum_a(tr['reward']),
        'length': alive.sum(0),
        'terminated': done.any(0).astype(float),
        'vel_err_mean': mean_a(tr['vel_err']),
        'vel_err_rms': np.sqrt(mean_a(tr['vel_err'] ** 2)),
        'yaw_err_mean_deg': np.rad2deg(mean_a(tr['yaw_err'])),
        'tilt_mean_deg': np.rad2deg(mean_a(tr['tilt'])),
        'tilt_max_deg': np.rad2deg(np.where(alive, np.nan_to_num(tr['tilt']), 0).max(0)),
        'angvel_max': np.where(alive, np.nan_to_num(tr['angvel']), 0).max(0),
        'servo_sat_frac': mean_a((np.abs(tr['servo']) >= 0.97 * rows['servo_max'][None, :, None]).any(-1)),
        'edf_sat_frac': mean_a(((tr['w'] > 0.98) | (tr['w'] < 0.02)).any(-1)),
    }
    for k in REWARD_TERMS:
        ep[f'r_{k}'] = sum_a(tr[f'r_{k}'])

    def stats(x):
        return dict(mean=float(np.mean(x)), median=float(np.median(x)), p10=float(np.percentile(x, 10)),
                    p90=float(np.percentile(x, 90)), min=float(np.min(x)), max=float(np.max(x)))

    summary = {k: stats(v) for k, v in ep.items()}
    print('\n' + '=' * 78)
    print(f'  {pol.description}')
    print('=' * 78)
    print(f'  termination rate     {summary["terminated"]["mean"] * 100:6.1f} %')
    print(f'  episode length       {summary["length"]["mean"]:6.1f} / {T}')
    print(f'  return               {summary["return"]["mean"]:+8.1f}  (median {summary["return"]["median"]:+.1f}, '
          f'p10 {summary["return"]["p10"]:+.1f})')
    print(f'  vel err  mean / rms  {summary["vel_err_mean"]["mean"]:.3f} / {summary["vel_err_rms"]["mean"]:.3f} m/s '
          f'(p90 {summary["vel_err_mean"]["p90"]:.3f})')
    print(f'  yaw err  mean        {summary["yaw_err_mean_deg"]["mean"]:.1f} deg (p90 {summary["yaw_err_mean_deg"]["p90"]:.1f})')
    print(f'  tilt mean / max      {summary["tilt_mean_deg"]["mean"]:.1f} / {summary["tilt_max_deg"]["mean"]:.1f} deg')
    print(f'  servo / EDF sat.     {summary["servo_sat_frac"]["mean"] * 100:.1f} % / {summary["edf_sat_frac"]["mean"] * 100:.1f} % of steps')
    print('=' * 78)

    out = {'policy': pol.description, 'configs': args.configs, 'episodes': n, 'steps': T,
           'summary': summary}

    if args.ai:
        print('\n[ai] reward terms (episode sums, mean over episodes):')
        for k in REWARD_TERMS:
            print(f'    {k:<10s} {summary[f"r_{k}"]["mean"]:+9.2f}')
        # per-config correlation with return
        corr = {}
        for k in SCALAR_PARAMS:
            if k in rows:
                x = np.asarray(rows[k], dtype=float)
                if x.std() > 0 and ep['return'].std() > 0:
                    corr[k] = float(np.corrcoef(x, ep['return'])[0, 1])
        for k, v in (('com_lat', np.linalg.norm(rows['arm'][:, :2], axis=1)),
                     ('arm_len', -rows['arm'][:, 2]),
                     ('tether_mag', np.linalg.norm(rows['tether_force'], axis=1))):
            if v.std() > 0 and ep['return'].std() > 0:
                corr[k] = float(np.corrcoef(v, ep['return'])[0, 1])
        if corr:
            print('[ai] correlation(param, return), sorted by |r|:')
            for k, v in sorted(corr.items(), key=lambda kv: -abs(kv[1])):
                print(f'    {k:<15s} {v:+.3f}')
        out['param_return_corr'] = corr

        worst = np.argsort(ep['return'])[: args.worst]
        out['worst'] = []
        for i in worst:
            info = {k: float(v[i]) for k, v in ep.items()}
            info['episode'] = int(i)
            info['params'] = {k: np.asarray(rows[k][i]).tolist() for k in rows}
            out['worst'].append(info)
            p = info['params']
            print(f'\n[ai] worst episode {i}: return {info["return"]:+.1f} len {int(info["length"])} '
                  f'term {int(info["terminated"])} vel_err {info["vel_err_mean"]:.2f} tilt_max {info["tilt_max_deg"]:.0f}deg')
            print(f'     mass {p["mass"]:.3f} tw {p["tw"]:.2f} arm {np.round(p["arm"], 4).tolist()} kq {p["kq"]:.3f} '
                  f'servo_max {np.rad2deg(p["servo_max"]):.1f}deg delay {p["act_delay"]} gust {p["gust_std"]:.2f} '
                  f'tether {np.round(p["tether_force"], 2).tolist()}')
            t_end = int(first_done[i]) if first_done[i] < T else T - 1
            print('     t     vx    vy    vz  | vrx   vry   vrz | tilt  yawE  |w|    | a0    a1    a2    a3   | srv0  srv1  | r')
            for t in range(max(0, t_end - args.tail), t_end + 1):
                v = tr['qvel'][t, i, :3]
                vr = tr['v_ref'][t, i]
                a = tr['action'][t, i]
                s = np.rad2deg(tr['servo'][t, i])
                print(f'   {t:4d} {v[0]:+.2f} {v[1]:+.2f} {v[2]:+.2f} | {vr[0]:+.2f} {vr[1]:+.2f} {vr[2]:+.2f} | '
                      f'{np.rad2deg(tr["tilt"][t, i]):4.0f} {np.rad2deg(tr["yaw_err"][t, i]):5.0f} {tr["angvel"][t, i]:5.1f} | '
                      f'{a[0]:+.2f} {a[1]:+.2f} {a[2]:+.2f} {a[3]:+.2f} | {s[0]:+5.1f} {s[1]:+5.1f} | {tr["reward"][t, i]:+.2f}')

        out_dir = Path(args.out or f'outputs/eval/{time.strftime("%Y-%m-%d/%H-%M-%S")}')
        out_dir.mkdir(parents=True, exist_ok=True)
        k_tr = min(args.trace, n)
        np.savez_compressed(out_dir / 'traces.npz',
                            **{k: v[:, :k_tr] for k, v in tr.items()},
                            **{f'param_{k}': np.asarray(v[:k_tr]) for k, v in rows.items()},
                            **{f'ep_{k}': v[:k_tr] for k, v in ep.items()})
        (out_dir / 'summary.json').write_text(json.dumps(out, indent=2))
        print(f'\n[ai] wrote {out_dir}/summary.json and traces.npz ({k_tr} episodes x {T} steps)')
    elif args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / 'summary.json').write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------- overlay
def _add_arrow(scn, start, vec, rgba, width=0.006):
    import mujoco
    if scn.ngeom >= scn.maxgeom or np.linalg.norm(vec) < 1e-4:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                        np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, start, start + vec)
    scn.ngeom += 1


def _add_capsule(scn, a, b, rgba, width):
    import mujoco
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, np.zeros(3), np.zeros(3), np.zeros(9),
                        np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CYLINDER, width, a, b)
    scn.ngeom += 1


def _draw_overlay(scn, s, d, p):
    """Appends nozzle, thrust, velocity reference, velocity and yaw reference geoms for env 0."""
    import jax.numpy as jnp
    from minidropper.utils.math_utils import quat_to_rotmat

    qpos = np.asarray(s.data.qpos[0])
    R = np.asarray(quat_to_rotmat(jnp.asarray(qpos[3:7])))
    com = qpos[:3]
    pivot = com + R @ p['arm']
    sv = d['servo']
    d_body = np.array([np.sin(sv[0]) * np.cos(sv[1]), -np.sin(sv[1]), np.cos(sv[0]) * np.cos(sv[1])])
    d_world = R @ d_body
    _add_capsule(scn, pivot, pivot - 0.035 * d_world, [0.2, 0.2, 0.25, 1], 0.024)
    _add_arrow(scn, pivot - 0.035 * d_world, -0.08 * float(d['thrust']) / (float(p['mass']) * 9.81) * d_world,
               [1.0, 0.5, 0.1, 0.9], 0.012)
    v = np.asarray(s.data.qvel[0, :3])
    vr = np.asarray(s.info['cmd']['v_ref'][0])
    _add_arrow(scn, com, 0.4 * vr, [0.1, 0.9, 0.2, 0.9])
    _add_arrow(scn, com, 0.4 * v, [0.2, 0.4, 1.0, 0.9])
    yr = float(s.info['cmd']['yaw_ref'][0])
    _add_arrow(scn, com + np.array([0, 0, 0.15]), 0.15 * np.array([np.cos(yr), np.sin(yr), 0.0]),
               [0.9, 0.9, 0.1, 0.9], 0.004)
    return R, v, vr, yr


# ------------------------------------------------------------------------------- GUI
def run_gui(args):
    import jax
    import jax.numpy as jnp
    import mujoco
    import mujoco.viewer

    from minidropper.utils.math_utils import yaw_from_rotmat

    env, venv, pol, rows = _build(args, 1)
    p = {k: np.asarray(v[0]) for k, v in rows.items()}
    print(f'[gui] {pol.description}')
    print(f'[gui] config "{args.configs}": mass {float(p["mass"]):.3f} kg, T/W {float(p["tw"]):.2f}, '
          f'arm {np.round(p["arm"], 3).tolist()}, delay {int(p["act_delay"])}')
    print(__doc__.split('Video')[0])

    reset = jax.jit(venv.reset)
    step = jax.jit(venv.step)
    act = jax.jit(pol.act)

    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)
    ui = {'manual': False, 'v': np.zeros(3), 'yaw': 0.0, 'reset': False}
    KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
    KEY_PGUP, KEY_PGDN, KEY_INSERT, KEY_BACKSPACE, KEY_HOME = 266, 267, 260, 259, 268

    def key_cb(key):
        dv = 0.25
        if key == KEY_UP: ui['v'][0] += dv
        elif key == KEY_DOWN: ui['v'][0] -= dv
        elif key == KEY_LEFT: ui['v'][1] += dv
        elif key == KEY_RIGHT: ui['v'][1] -= dv
        elif key == KEY_PGUP: ui['v'][2] += dv
        elif key == KEY_PGDN: ui['v'][2] -= dv
        elif key == ord(','): ui['yaw'] += np.deg2rad(15)
        elif key == ord('.'): ui['yaw'] -= np.deg2rad(15)
        elif key == KEY_HOME: ui['v'][:] = 0.0
        elif key == KEY_INSERT:
            ui['manual'] = not ui['manual']
            print(f'[gui] command mode: {"MANUAL" if ui["manual"] else "random"}')
            return
        elif key == KEY_BACKSPACE:
            ui['reset'] = True
            return
        else:
            return
        ui['manual'] = True
        ui['v'] = np.clip(ui['v'], [-1, -1, -0.5], [1, 1, 0.5])
        ui['yaw'] = float(np.clip(ui['yaw'], -1.5, 1.5))
        print(f'[gui] manual cmd v = {np.round(ui["v"], 2).tolist()} m/s, yaw = {np.rad2deg(ui["yaw"]):.0f} deg')

    key = jax.random.PRNGKey(args.seed)
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_cb) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env.unwrapped._body_id
        viewer.cam.distance = 1.2
        episode = 0
        while viewer.is_running():
            key, k = jax.random.split(key)
            s = reset(k[None])
            h = pol.init_carry(1)
            ui['reset'] = False
            ep_ret, t = 0.0, 0
            terms = {kk: 0.0 for kk in REWARD_TERMS}
            episode += 1
            while viewer.is_running() and not ui['reset']:
                t0 = time.time()
                if ui['manual']:
                    cmd = dict(s.info['cmd'])
                    cmd['v_target'] = jnp.asarray(ui['v'], dtype=jnp.float32)[None]
                    cmd['yaw_target'] = jnp.asarray([ui['yaw']], dtype=jnp.float32)
                    cmd['timer'] = jnp.asarray([1e9], dtype=jnp.float32)
                    s = s.replace(info={**s.info, 'cmd': cmd})
                a, h = act(s.obs, h)
                s = step(s, a)
                t += 1
                r = float(s.reward[0])
                ep_ret += r
                d = {kk: np.asarray(v[0]) for kk, v in s.info['diag'].items()}
                for kk in REWARD_TERMS:
                    terms[kk] += float(d[f'r_{kk}'])

                mj_data.qpos[:] = np.asarray(s.data.qpos[0])
                mj_data.qvel[:] = np.asarray(s.data.qvel[0])
                mujoco.mj_forward(mj_model, mj_data)

                viewer.user_scn.ngeom = 0
                R, v, vr, yr = _draw_overlay(viewer.user_scn, s, d, p)
                sv = d['servo']
                viewer.sync()

                if t % args.print_every == 0 or float(s.done[0]) > 0:
                    yaw = float(yaw_from_rotmat(jnp.asarray(R)))
                    print(f'ep {episode} t {t * env.dt:5.2f}s | v {np.round(v, 2)} ref {np.round(vr, 2)} | '
                          f'yaw {np.rad2deg(yaw):+5.0f} ref {np.rad2deg(yr):+5.0f} | tilt {np.rad2deg(float(d["tilt"])):4.1f} | '
                          f'w {np.round(d["w"], 2)} srv {np.round(np.rad2deg(sv), 1)} | r {r:+.2f} R {ep_ret:+.1f}')
                if float(s.done[0]) > 0 or t >= (args.steps or 10_000):
                    why = 'TERMINATED' if float(s.done[0]) > 0 else 'time limit'
                    print(f'[gui] episode {episode} {why} after {t} steps, return {ep_ret:+.1f}; terms ' +
                          ' '.join(f'{kk} {vv:+.1f}' for kk, vv in terms.items()))
                    break
                time.sleep(max(0.0, env.dt / args.speed - (time.time() - t0)))


# ----------------------------------------------------------------------------- video
def run_video(args):
    import imageio
    import jax
    import mujoco

    env, venv, pol, rows = _build(args, 1)
    p = {k: np.asarray(v[0]) for k, v in rows.items()}
    T = args.steps or 1000
    skip = max(1, round(1.0 / (env.dt * args.fps)))
    fps = 1.0 / (env.dt * skip)
    out = Path(args.out or f'outputs/videos/{time.strftime("%Y-%m-%d_%H-%M-%S")}.mp4')
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f'[video] {pol.description}\n[video] {T} steps ({T * env.dt:.1f} s) on config "{args.configs}" '
          f'-> {out} at {fps:.1f} fps, {args.width}x{args.height}')

    reset = jax.jit(venv.reset)
    step = jax.jit(venv.step)
    act = jax.jit(pol.act)

    mj_model = env.mj_model
    mj_model.vis.global_.offwidth = max(mj_model.vis.global_.offwidth, args.width)
    mj_model.vis.global_.offheight = max(mj_model.vis.global_.offheight, args.height)
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(mj_model, cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = env.unwrapped._body_id
    cam.distance = 1.2

    key = jax.random.PRNGKey(args.seed)
    key, k = jax.random.split(key)
    s = reset(k[None])
    h = pol.init_carry(1)
    episodes, ep_ret = 1, 0.0
    with imageio.get_writer(out, fps=fps, codec='libx264', quality=8, macro_block_size=8) as writer:
        for t in range(T):
            a, h = act(s.obs, h)
            s = step(s, a)
            ep_ret += float(s.reward[0])
            if t % skip == 0:
                d = {kk: np.asarray(v[0]) for kk, v in s.info['diag'].items()}
                mj_data.qpos[:] = np.asarray(s.data.qpos[0])
                mj_data.qvel[:] = np.asarray(s.data.qvel[0])
                mujoco.mj_forward(mj_model, mj_data)
                renderer.update_scene(mj_data, camera=cam)
                _draw_overlay(renderer.scene, s, d, p)
                writer.append_data(renderer.render())
            if float(s.done[0]) > 0:
                print(f'[video] episode {episodes} TERMINATED at step {t}, return {ep_ret:+.1f}; resetting')
                key, k = jax.random.split(key)
                s = reset(k[None])
                h = pol.init_carry(1)
                episodes, ep_ret = episodes + 1, 0.0
    renderer.close()
    print(f'[video] wrote {out} ({episodes} episode(s), last return {ep_ret:+.1f})')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--policy', required=True, help='trained_model.pkl | chunk_XXX.pkl | student_*.pkl')
    ap.add_argument('--mode', choices=['gui', 'batch', 'video'], default='batch')
    ap.add_argument('--configs', '--config', default='test')
    ap.add_argument('--teacher_id', type=int, default=None)
    ap.add_argument('--params_table', default='data/params/v1.npz')
    ap.add_argument('--episodes', type=int, default=512)
    ap.add_argument('--steps', type=int, default=None,
                    help='batch: episode steps (500); gui: max steps; video: total steps (1000)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ai', action='store_true', help='verbose diagnostics + dumps')
    ap.add_argument('--out', default=None)
    ap.add_argument('--trace', type=int, default=32, help='--ai: episodes with full per-step traces')
    ap.add_argument('--worst', type=int, default=3)
    ap.add_argument('--tail', type=int, default=15, help='--ai: steps printed before termination')
    ap.add_argument('--print_every', type=int, default=25)
    ap.add_argument('--speed', type=float, default=1.0, help='gui: real-time factor')
    ap.add_argument('--fps', type=float, default=30.0, help='video: target frame rate')
    ap.add_argument('--width', type=int, default=960, help='video: frame width')
    ap.add_argument('--height', type=int, default=720, help='video: frame height')
    args = ap.parse_args()
    if args.mode in ('gui', 'video') and args.configs == 'test':
        args.configs = 'nominal'
    if args.mode == 'gui':
        run_gui(args)
    elif args.mode == 'video':
        os.environ.setdefault('MUJOCO_GL', 'egl')
        run_video(args)
    else:
        run_batch(args)


if __name__ == '__main__':
    main()
