"""Load any trained policy behind one batched, recurrent-capable interface.

    pol = load_policy(path, teacher_id=None)
    h = pol.init_carry(batch)
    action, h = pol.act(obs_dict, h)        # batched obs, deterministic

Supported checkpoints:
    <run>/trained_model.pkl   (train_single.py, needs policy_meta.json next to it)
    <run>/chunk_XXX.pkl       (train_multi.py, pick a teacher with teacher_id)
    <run>/student_*.pkl       (student/train.py)
"""

import dataclasses
import json
from pathlib import Path
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from brax.io import model
from brax.training.acme import running_statistics

from minidropper.teachers.training import networks as ppo_networks


@dataclasses.dataclass
class Policy:
    kind: str
    description: str
    act: Callable            # (obs_dict, carry) -> (action, carry)
    init_carry: Callable     # batch -> carry
    config_row: Optional[dict] = None   # vehicle config the policy was trained on (teachers)
    env_cfg: Optional[dict] = None


def _ppo_policy(params, network_kwargs, obs_shape, kind, description, config_row=None):
    net = ppo_networks.make_ppo_networks(
        obs_shape, 4, preprocess_observations_fn=running_statistics.normalize, **network_kwargs)
    make_policy = ppo_networks.make_inference_fn(net)
    pi = make_policy((params[0], params[1]), deterministic=True)

    def act(obs, carry):
        a, _ = pi(obs, jax.random.PRNGKey(0))
        return a, carry

    return Policy(kind, description, act, lambda b: jnp.zeros((b, 1)), config_row)


def peek_env_cfg(path) -> dict:
    """Env config a checkpoint was trained with (needed to build matching obs)."""
    path = Path(path)
    for name in ('policy_meta.json', 'teachers_meta.json'):
        if (path.parent / name).exists():
            return json.loads((path.parent / name).read_text()).get('env') or {}
    blob = model.load_params(str(path))
    if isinstance(blob, dict):
        return blob.get('env_cfg') or {}
    return {}


def load_policy(path, obs_shape, teacher_id: Optional[int] = None) -> Policy:
    path = Path(path)
    blob = model.load_params(str(path))

    if isinstance(blob, dict) and blob.get('kind') == 'gru_student':
        from minidropper.student.training import dagger
        from ml_collections import config_dict
        student = dagger.make_student(config_dict.ConfigDict(blob['student_cfg']))
        sparams = jax.tree_util.tree_map(jnp.asarray, blob['params'])
        norm = {k: jnp.asarray(v) for k, v in blob['normalizer'].items()}

        def act(obs, h):
            h, a = dagger.student_step(student, sparams, norm, h, obs['student_obs'])
            return a, h

        return Policy('student', f'GRU student {path}', act,
                      lambda b: jnp.zeros((b, student.hidden_dim)), env_cfg=blob.get('env_cfg'))

    if isinstance(blob, dict) and 'teacher_ids' in blob:
        meta = json.loads((path.parent / 'teachers_meta.json').read_text())
        ids = list(np.asarray(blob['teacher_ids']))
        if teacher_id is None:
            teacher_id = int(ids[int(np.argmax(blob['eval']['return']))])
        if teacher_id not in ids:
            raise ValueError(f'teacher {teacher_id} not in {path.name}: {ids}')
        j = ids.index(teacher_id)
        params = jax.tree_util.tree_map(lambda x: jnp.asarray(x[j]), blob['params'])
        row = {k: v[j] for k, v in blob['config_rows'].items()}
        pol = _ppo_policy(params, meta['network'], obs_shape, 'teacher',
                          f'teacher {teacher_id} ({path})', row)
        pol.env_cfg = meta.get('env')
        return pol

    meta_path = path.parent / 'policy_meta.json'
    meta = json.loads(meta_path.read_text())
    n = meta['network']
    kwargs = dict(
        policy_dec_hidden_layers=n['policy_dec_hidden_layers'],
        action_hidden_layer_sizes=n['action_hidden_layers'],
        value_hidden_layer_sizes=n['value_hidden_layers'],
        activation=n['activation'], policy_obs_key=n['policy_obs_key'],
        value_obs_key=n['value_obs_key'])
    params = jax.tree_util.tree_map(jnp.asarray, blob)
    pol = _ppo_policy(params, kwargs, obs_shape, 'ppo', f'PPO ({meta["mode"]}) {path}')
    pol.env_cfg = meta.get('env')
    return pol
