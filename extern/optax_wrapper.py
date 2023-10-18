"""OptaxWrapper snippets of code copied from KFAC-JAX for ease of modification."""

# Copied from https://github.com/deepmind/kfac-jax/blob/main/examples/optimizers.py
from typing import Any, Callable, Mapping, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import kfac_jax
import optax


class OptaxWrapper:
    """Wrapper class for Optax optimizers to have the same interface as KFAC."""
    OptaxState = Any

    def __init__(
        self,
        value_and_grad_func: kfac_jax.optimizer.ValueAndGradFunc,
        value_func_has_aux: bool,
        value_func_has_state: bool,
        value_func_has_rng: bool,
        optax_optimizer: optax.GradientTransformation,
        batch_process_func: Optional[Callable[[Any], Any]] = lambda x: x,
    ):
        """Initializes the Optax wrapper.
        Args:
        value_and_grad_func: Python callable. The function should return the value
            of the loss to be optimized and its gradients. If the argument
            `value_func_has_aux` is `False` then the interface should be:
            loss, loss_grads = value_and_grad_func(params, batch)
            If `value_func_has_aux` is `True` then the interface should be:
            (loss, aux), loss_grads = value_and_grad_func(params, batch)
        value_func_has_aux: Boolean. Specifies whether the provided callable
            `value_and_grad_func` returns the loss value only, or also some
            auxiliary data. (Default: `False`)
        value_func_has_state: Boolean. Specifies whether the provided callable
            `value_and_grad_func` has a persistent state that is inputted and it
            also outputs an update version of it. (Default: `False`)
        value_func_has_rng: Boolean. Specifies whether the provided callable
            `value_and_grad_func` additionally takes as input an rng key. (Default:
            `False`)
        optax_optimizer: The optax optimizer to be wrapped.
        batch_process_func: Callable. A function which to be called on each batch
            before feeding to the KFAC on device. This could be useful for specific
            device input optimizations. (Default: `lambda x: x`)
        """
        self._value_and_grad_func = value_and_grad_func
        self._value_func_has_aux = value_func_has_aux
        self._value_func_has_state = value_func_has_state
        self._value_func_has_rng = value_func_has_rng
        self._optax_optimizer = optax_optimizer
        self._batch_process_func = batch_process_func or (lambda x: x)
        # CHANGE: Don't try to pmap the step
        # self._jit_step = jax.pmap(
        #     self._step, axis_name="optax_axis", donate_argnums=[0, 1, 2, 3, 5])
        self._jit_step = jax.jit(self._step)

    def init(
        self,
        params: kfac_jax.utils.Params,
        rng: jnp.ndarray,
        batch: kfac_jax.utils.Batch,
        func_state: Optional[kfac_jax.utils.FuncState] = None
    ) -> OptaxState:
        """Initializes the optimizer and returns the appropriate optimizer state."""
        del rng, batch, func_state
        # CHANGE: Don't try to pmap the optimiser initialiser
        # return jax.pmap(self._optax_optimizer.init)(params)
        return self._optax_optimizer.init(params)

    def _step(
        self,
        params: kfac_jax.utils.Params,
        state: OptaxState,
        rng: chex.PRNGKey,
        batch: kfac_jax.utils.Batch,
        func_state: Optional[kfac_jax.utils.FuncState] = None,
    ) -> kfac_jax.optimizer.FuncOutputs:
        """A single step of optax."""
        batch = self._batch_process_func(batch)
        func_args = kfac_jax.optimizer.make_func_args(
            params, func_state, rng, batch,
            has_state=self._value_func_has_state,
            has_rng=self._value_func_has_rng
        )
        out, grads = self._value_and_grad_func(*func_args)

        if not self._value_func_has_aux and not self._value_func_has_state:
            loss, new_func_state, stats = out, None, {}
        else:
            loss, other = out
            if self._value_func_has_aux and self._value_func_has_state:
                new_func_state, stats = other
            elif self._value_func_has_aux:
                new_func_state, stats = None, other
            else:
                new_func_state, stats = other, {}
        stats["loss"] = loss
        # CHANGE: No need to mean-reduce over parallel devices
        # stats, grads = jax.lax.pmean((stats, grads), axis_name="optax_axis")
        # Compute and apply updates via our optimizer.
        updates, new_state = self._optax_optimizer.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)

        # Add batch size
        # CHANGE: Update deprecated reference to jax.tree_leaves
        batch_size = jax.tree_util.tree_leaves(batch)[0].shape[0]
        stats["batch_size"] = batch_size * jax.device_count()
        stats['gradient_norm'] = jnp.linalg.norm(jax.flatten_util.ravel_pytree(grads)[0])
        stats['update_norm'] = jnp.linalg.norm(jax.flatten_util.ravel_pytree(updates)[0])

        if self._value_func_has_state:
            return new_params, new_state, new_func_state, stats
        else:
            return new_params, new_state, stats

    def step(
        self,
        params: kfac_jax.utils.Params,
        state: OptaxState,
        rng: jnp.ndarray,
        # data_iterator: Iterator[kfac_jax.utils.Batch],
        batch: kfac_jax.utils.Batch,
        func_state: Optional[kfac_jax.utils.FuncState] = None,
        global_step_int: Optional[int] = None
    ) -> Union[Tuple[kfac_jax.utils.Params, Any, kfac_jax.utils.FuncState,
                     Mapping[str, jnp.ndarray]],
               Tuple[kfac_jax.utils.Params, Any,
                     Mapping[str, jnp.ndarray]]]:
        """A step with similar interface to KFAC."""
        result = self._jit_step(
            params=params,
            state=state,
            rng=rng,
            # batch=next(data_iterator),
            # CHANGE: Using batch instead of data_iterator here (and in arguments)
            batch=batch,
            func_state=func_state,
        )
        step = jnp.asarray(global_step_int + 1)
        step = kfac_jax.utils.replicate_all_local_devices(step)
        result[-1]["step"] = step
        result[-1]["data_seen"] = step * result[-1]["batch_size"]

        return result
