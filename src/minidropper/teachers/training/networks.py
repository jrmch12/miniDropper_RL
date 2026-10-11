"""PPO networks (copied from arcdrone priviledged_landing_rl/training/networks.py).

Observation dict keys (see minidropper/task/obs.py):
    policy_obs   : teacher obs (student obs + EDF speeds + servo angles)
    value_obs    : critic obs
    student_obs  : deployable obs (used as policy input for the DR baselines)
    student_hist : stacked deployable obs (MLP+history DR baseline)

Architecture:
    Policy : obs[policy_obs_key] -> MLP decoder -> MLP action head -> NormalTanh params
    Value  : obs[value_obs_key]  -> MLP -> scalar
"""

from typing import Any, Callable, Mapping, Sequence, Tuple

from brax.training import distribution
from brax.training import types
from brax.training.networks import normalizer_select
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp

from minidropper.common.networks import ACTIVATIONS, MLP


@flax.struct.dataclass
class FeedForwardNetwork:
    init: Callable[..., Any]
    apply: Callable[..., Any]


@flax.struct.dataclass
class PPONetworks:
    """Brax-PPO-compatible network container."""
    policy_network: FeedForwardNetwork
    value_network: FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def _shape_last_dim(shape_spec: Any) -> int:
    if isinstance(shape_spec, int):
        return shape_spec
    return int(jax.tree_util.tree_flatten(shape_spec)[0][-1])


def make_ppo_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = (
        types.identity_observation_preprocessor
    ),
    policy_dec_hidden_layers: Sequence[int] = (64, 64),
    action_hidden_layer_sizes: Sequence[int] = (64,),
    value_hidden_layer_sizes: Sequence[int] = (256, 256, 256),
    activation: str = 'tanh',
    policy_obs_key: str = 'policy_obs',
    value_obs_key: str = 'value_obs',
) -> PPONetworks:
    """Build flat-MLP policy + value PPO networks."""

    act_fn = ACTIVATIONS[activation] if isinstance(activation, str) else activation
    kernel_init = jax.nn.initializers.lecun_uniform()

    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    policy_decoder = MLP(
        layer_sizes=list(policy_dec_hidden_layers),
        activation=act_fn,
        kernel_init=kernel_init,
        activate_final=True,
    )
    policy_action_head = MLP(
        layer_sizes=list(action_hidden_layer_sizes)
            + [parametric_action_distribution.param_size],
        activation=act_fn,
        kernel_init=kernel_init,
        activate_final=False,
    )
    value_mlp = MLP(
        layer_sizes=list(value_hidden_layer_sizes) + [1],
        activation=act_fn,
        kernel_init=kernel_init,
        activate_final=False,
    )

    if isinstance(observation_size, Mapping):
        policy_obs_dim = _shape_last_dim(observation_size[policy_obs_key])
        value_obs_dim = _shape_last_dim(observation_size[value_obs_key])
    else:
        policy_obs_dim = value_obs_dim = _shape_last_dim(observation_size)

    dummy_policy_obs = jnp.zeros((1, policy_obs_dim))
    dummy_value_obs = jnp.zeros((1, value_obs_dim))

    def _preprocess(obs, pparams, key):
        if isinstance(obs, Mapping):
            return preprocess_observations_fn(obs[key], normalizer_select(pparams, key))
        return preprocess_observations_fn(obs, pparams)

    def _value_init(key):
        return value_mlp.init(key, dummy_value_obs)

    def _value_apply(pparams, params, obs):
        return jnp.squeeze(
            value_mlp.apply(params, _preprocess(obs, pparams, value_obs_key)), axis=-1)

    def _policy_init(key):
        k1, k2 = jax.random.split(key)
        dec_params = policy_decoder.init(k1, dummy_policy_obs)
        feats = policy_decoder.apply(dec_params, dummy_policy_obs)
        head_params = policy_action_head.init(k2, feats)
        return dec_params, head_params

    def _policy_apply(pparams, params, obs):
        feats = policy_decoder.apply(params[0], _preprocess(obs, pparams, policy_obs_key))
        return policy_action_head.apply(params[1], feats)

    return PPONetworks(
        policy_network=FeedForwardNetwork(init=_policy_init, apply=_policy_apply),
        value_network=FeedForwardNetwork(init=_value_init, apply=_value_apply),
        parametric_action_distribution=parametric_action_distribution,
    )


def make_inference_fn(ppo_networks: PPONetworks, compute_value: bool = False):
    """Policy factory compatible with brax PPO (params = (normalizer, policy, value))."""

    def make_policy(params: types.Params, deterministic: bool = False) -> types.Policy:

        def policy(observations: types.Observation,
                   key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
            logits = ppo_networks.policy_network.apply(params[0], params[1], observations)
            if deterministic:
                return ppo_networks.parametric_action_distribution.mode(logits), {}
            dist = ppo_networks.parametric_action_distribution
            raw_actions = dist.sample_no_postprocessing(logits, key_sample)
            log_prob = dist.log_prob(logits, raw_actions)
            extras = {
                'log_prob': log_prob,
                'raw_action': raw_actions,
                'distribution_params': logits,
            }
            if compute_value:
                extras['value'] = ppo_networks.value_network.apply(
                    params[0], params[2], observations)
            return dist.postprocess(raw_actions), extras

        return policy

    return make_policy
