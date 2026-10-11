"""Shared neural-network building blocks."""

from typing import Any, Callable, Sequence

from flax import linen as nn
import jax
import jax.numpy as jnp


ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]

ACTIVATIONS = {'tanh': nn.tanh, 'relu': nn.relu, 'swish': nn.swish, 'elu': nn.elu}


class MLP(nn.Module):
  layer_sizes: Sequence[int]
  activation: ActivationFn = nn.tanh
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray):
    for i, size in enumerate(self.layer_sizes):
      x = nn.Dense(
          size,
          kernel_init=self.kernel_init,
          name=f'hidden_{i}',
      )(x)
      if i != len(self.layer_sizes) - 1 or self.activate_final:
        x = self.activation(x)
    return x


class GRUPolicy(nn.Module):
  """RAPTOR student: Dense -> GRU -> Dense, one step at a time.

  __call__(h, x) -> (h_next, action_pre_tanh)
  """

  hidden_dim: int = 64
  input_dim: int = 64
  action_size: int = 4

  @nn.compact
  def __call__(self, h: jnp.ndarray, x: jnp.ndarray):
    x = nn.relu(nn.Dense(self.input_dim, name='input')(x))
    h, y = nn.GRUCell(features=self.hidden_dim, name='gru')(h, x)
    return h, nn.Dense(self.action_size, name='output')(y)
