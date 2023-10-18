"""Source: https://github.com/deepmind/optax/blob/4afbd1ee320f61f908bbf4f853e8cbe2a18c102c/optax/_src/transform.py#L307"""
from optax._src.transform import *


def scale_by_damped_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  """Rescale updates according to the Adam algorithm.
  References:
    [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)
  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.
  Returns:
    A `GradientTransformation` object.
  """

  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = jax.tree_util.tree_map(  # First moment
        lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
    nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
    return dict(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, eps=jnp.array(eps))

  def update_fn(updates, state, params=None):
    del params
    mu = update_moment(updates, state['mu'], b1, 1)
    nu = update_moment_per_elem_norm(updates, state['nu'], b2, 2)
    count_inc = numerics.safe_int32_increment(state['count'])
    mu_hat = bias_correction(mu, b1, count_inc)
    nu_hat = bias_correction(nu, b2, count_inc)
    updates = jax.tree_util.tree_map(
        lambda m, v: m / (jnp.sqrt(v + eps_root) + state['eps']), mu_hat, nu_hat)
    mu = utils.cast_tree(mu, mu_dtype)
    return updates, dict(count=count_inc, mu=mu, nu=nu, eps=state['eps'])

  return base.GradientTransformation(init_fn, update_fn)
