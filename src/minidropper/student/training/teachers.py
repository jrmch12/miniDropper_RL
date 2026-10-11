"""Loading a multi-teacher run and evaluating stacked teachers."""

import csv
import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from brax.io import model
from brax.training.acme import running_statistics

from minidropper.dr import sample_params as sp
from minidropper.teachers.training import networks as ppo_networks


def load_teachers(teachers_dirs, only_accepted=True):
    """Stack teachers from one or more train_multi.py run dirs.

    Returns dict(ids, config_rows, params=(normalizer, policy, value) stacked, meta, index_rows).
    """
    if isinstance(teachers_dirs, (str, Path)):
        teachers_dirs = [teachers_dirs]
    ids, rows, params, index_rows, meta = [], [], [], [], None
    for d in teachers_dirs:
        d = Path(d)
        meta = json.loads((d / 'teachers_meta.json').read_text())
        with open(d / 'teachers_index.csv') as f:
            idx = {int(r['teacher_id']): r for r in csv.DictReader(f)}
        for chunk in sorted(d.glob('chunk_*.pkl')):
            c = model.load_params(str(chunk))
            for j, tid in enumerate(c['teacher_ids']):
                r = idx[int(tid)]
                if only_accepted and not int(r['accepted']):
                    continue
                ids.append(int(tid))
                rows.append({k: v[j] for k, v in c['config_rows'].items()})
                params.append(jax.tree_util.tree_map(lambda x: x[j], c['params']))
                index_rows.append(r)
    if not ids:
        raise RuntimeError(f'No teachers found in {teachers_dirs} (only_accepted={only_accepted})')
    stacked = jax.tree_util.tree_map(lambda *xs: np.stack(xs), *params)
    return dict(ids=np.array(ids), config_rows=sp.stack_configs(rows), params=stacked,
                meta=meta, index_rows=index_rows)


def make_teacher_network(meta, obs_shape, action_size=4):
    return ppo_networks.make_ppo_networks(
        obs_shape, action_size, preprocess_observations_fn=running_statistics.normalize,
        **meta['network'])


def teacher_action_fn(teacher_net, n_teachers):
    """Deterministic actions of N stacked teachers for obs batched teacher-major."""
    dist = teacher_net.parametric_action_distribution

    def act(tparams, obs):
        norm, pol = tparams[0], tparams[1]
        obs_k = jax.tree_util.tree_map(
            lambda x: x.reshape((n_teachers, -1) + x.shape[1:]), obs)

        def one(n, p, o):
            return dist.mode(teacher_net.policy_network.apply(n, p, o))

        a = jax.vmap(one)(norm, pol, obs_k)
        return a.reshape((-1,) + a.shape[2:])

    return act
