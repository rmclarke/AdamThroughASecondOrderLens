from copy import deepcopy
from functools import partial

import kfac_jax
import jax
import jax.numpy as jnp
import optax

import util


def correction_line_search_by_jvp(value_and_grad_fn,
                                  parameter_gradient,
                                  correction_gradient,
                                  kfac_estimator_state,
                                  model_params,
                                  model_state,
                                  batch,
                                  direction,
                                  damping):
    flat_kfac_estimator_state, kfac_estimator_state_treedef = jax.tree_util.tree_flatten(kfac_estimator_state)
    _, hessian_direction = jax.jvp(
        lambda *kfac_es: value_and_grad_fn(
            parameter_gradient,
            jax.tree_util.tree_unflatten(kfac_estimator_state_treedef, kfac_es),
            model_params,
            model_state,
            batch,
            damping)[1],
        flat_kfac_estimator_state,
        jax.tree_util.tree_flatten(direction)[0])

    ravelled_gradient, _ = jax.flatten_util.ravel_pytree(correction_gradient)
    ravelled_direction, _ = jax.flatten_util.ravel_pytree(direction)
    ravelled_hessian_direction, _ = jax.flatten_util.ravel_pytree(
        hessian_direction)
    step_size = -((ravelled_gradient.T @ ravelled_direction)
                  / (ravelled_direction.T @ ravelled_hessian_direction
                     + damping * ravelled_direction.T @ ravelled_direction))
    return step_size


def learning_rate_from_hessian(value_and_grad_fn,
                               func_params,
                               func_state,
                               gradient,
                               direction,
                               batch,
                               rng,
                               damping=0,
                               saddle_free_lr_curvature=False,
                               lr_curvature_ema=0,
                               last_curvature=None):
    flat_params, params_treedef = jax.tree_util.tree_flatten(func_params)
    _, hessian_direction = jax.jvp(
        lambda *p: value_and_grad_fn(
            jax.tree_util.tree_unflatten(params_treedef, p),
            func_state,
            rng,
            batch)[1],
        flat_params,
        jax.tree_util.tree_flatten(direction)[0])

    ravelled_gradient, _ = jax.flatten_util.ravel_pytree(gradient)
    ravelled_direction, _ = jax.flatten_util.ravel_pytree(direction)
    ravelled_hessian_direction, _ = jax.flatten_util.ravel_pytree(
        hessian_direction)
    second_order_model_term = (ravelled_direction.T @ ravelled_hessian_direction
                               + damping * ravelled_direction.T @ ravelled_direction)
    if lr_curvature_ema:
        second_order_model_term = lr_curvature_ema*last_curvature + (1-lr_curvature_ema)*second_order_model_term
    grad_alignment = ravelled_gradient.T @ ravelled_direction
    if saddle_free_lr_curvature:
        step_size = -grad_alignment / jnp.abs(second_order_model_term)
    else:
        step_size = -grad_alignment / second_order_model_term
    return step_size, second_order_model_term, grad_alignment


def learning_rate_from_fisher(model_fn,
                              loss_name,
                              func_params,
                              func_state,
                              gradient,
                              direction,
                              batch,
                              rng,
                              damping=0,
                              saddle_free_lr_curvature=False,
                              lr_curvature_ema=0,
                              last_curvature=None):
    flat_params, params_treedef = jax.tree_util.tree_flatten(func_params)
    model_pass = lambda *p: model_fn(
        jax.tree_util.tree_unflatten(params_treedef, p),
        func_state,
        rng,
        batch)
    model_output, jacobian_direction, _ = jax.jvp(
        model_pass,
        flat_params,
        jax.tree_util.tree_flatten(direction)[0],
        has_aux=True)

    match loss_name:
        case 'cross_entropy_loss':
            output_probabilities = jax.nn.softmax(model_output, axis=-1)
            batched_inner_product = jnp.einsum('...a, ...a -> ...',
                                               output_probabilities,
                                               jacobian_direction)
            fisher_jacobian_vector_product = output_probabilities * (jacobian_direction - batched_inner_product[..., None])
        case 'mse_loss' | 'null_loss':
            # loss_fisher is identity
            fisher_jacobian_vector_product = jacobian_direction
        case _: raise ValueError(f'Unknown loss {loss_name}')
    curvature_term = jnp.einsum('...a, ...a -> ...',
                                jacobian_direction,
                                fisher_jacobian_vector_product)
    avg_curvature_term = jnp.nanmean(curvature_term)

    ravelled_gradient, _ = jax.flatten_util.ravel_pytree(gradient)
    ravelled_direction, _ = jax.flatten_util.ravel_pytree(direction)
    second_order_model_term = (avg_curvature_term
                               + damping * ravelled_direction.T @ ravelled_direction)
    if lr_curvature_ema:
        second_order_model_term = lr_curvature_ema*last_curvature + (1-lr_curvature_ema)*second_order_model_term
    grad_alignment = (ravelled_gradient.T @ ravelled_direction)
    if saddle_free_lr_curvature:
        step_size = -grad_alignment / jnp.abs(second_order_model_term)
    else:
        step_size = -grad_alignment / second_order_model_term
    return step_size, second_order_model_term, grad_alignment


class WrappedQuadraticLROptimiser():

    def __init__(self,
                 value_and_grad_fn,
                 model_fn,
                 loss_name,
                 curvature_matrix,
                 damping_curvature='step_size',
                 damping_algorithm='kfac',
                 initial_damping=0,
                 damping_decrease_factor=None,
                 damping_increase_factor=None,
                 damping_clipping=(None, None),
                 update_amplification=1,
                 gradient_norm_clipping=None,
                 direction_clipping=None,
                 update_norm_clipping=None,
                 lr_clipping=None,
                 lr_ema=0,
                 saddle_free_lr_curvature=False,
                 lr_curvature_ema=0,
                 scaling_envelope_peak=None,
                 total_steps=None,
                 weight_decay=0,
                 **base_kwargs):
        self.value_and_grad_fn = value_and_grad_fn
        self.model_fn = model_fn
        self.loss_name = loss_name
        self.curvature_matrix = curvature_matrix
        self.damping_curvature = damping_curvature
        self.base_init, self.base_update = self.base_optimiser(**base_kwargs)

        self.initial_damping = initial_damping
        self.damping_decrease_factor = damping_decrease_factor
        self.damping_increase_factor = damping_increase_factor
        self.damping_clipping = damping_clipping
        self.damping_algorithm = damping_algorithm
        self.update_amplification = update_amplification
        self.gradient_norm_clipping = gradient_norm_clipping
        self.direction_clipping = direction_clipping
        self.update_norm_clipping = update_norm_clipping
        self.lr_clipping = lr_clipping
        self.lr_ema = lr_ema
        self.saddle_free_lr_curvature = saddle_free_lr_curvature
        self.lr_curvature_ema = lr_curvature_ema
        self.weight_decay = weight_decay
        if scaling_envelope_peak:
            self.envelope_schedule = util.linear_warmup_cosine_decay_schedule(
                peak_steps=scaling_envelope_peak,
                end_steps=total_steps)
        else:
            self.envelope_schedule = lambda _: 1.0

    @property
    def base_optimiser(self):
        pass

    def init(self, model_params, *args, **kwargs):
        del args, kwargs
        return dict(base=self.base_init(model_params),
                    damping=self.initial_damping,
                    lr_ema=jnp.array(0., dtype=jnp.float32),
                    last_curvature=jnp.array(1e7, dtype=jnp.float32))

    @partial(jax.jit, static_argnums=(0,))
    @util.except_ray_jit_only_once
    def step(self,
             params,
             optimiser_state,
             batch,
             func_state,
             rng,
             global_step_int,
             *args,
             **kwargs):
        sub_rng1, sub_rng2, sub_rng3 = jax.random.split(rng, 3)
        del args, kwargs, rng
        new_optimiser_state = dict(damping=optimiser_state['damping'],
                                   lr_ema=optimiser_state['lr_ema'])
        ((loss, (new_func_state, statistics)),
         gradient) = self.value_and_grad_fn(params, func_state, sub_rng1, batch)
        if self.gradient_norm_clipping:
            gradient_norm = jnp.linalg.norm(jax.flatten_util.ravel_pytree(gradient)[0])
            gradient = jax.lax.cond(gradient_norm > self.gradient_norm_clipping,
                                    lambda grad: jax.tree_util.tree_map(
                                        lambda g: g * self.gradient_norm_clipping / gradient_norm,
                                        grad),
                                    lambda grad: grad,
                                    gradient)
        direction, new_optimiser_state['base'] = self.base_update(gradient,
                                                                  optimiser_state['base'],
                                                                  batch)
        if self.direction_clipping:
            direction = jax.tree_util.tree_map(
                lambda dir: jnp.clip(dir, a_min=-self.direction_clipping, a_max=self.direction_clipping),
                direction)
        step_size_kwargs = dict(func_params=params,
                                func_state=func_state,
                                gradient=gradient,
                                direction=direction,
                                batch=batch,
                                damping=optimiser_state['damping'],
                                rng=sub_rng2,
                                saddle_free_lr_curvature=self.saddle_free_lr_curvature,
                                lr_curvature_ema=self.lr_curvature_ema,
                                last_curvature=optimiser_state['last_curvature'])
        match self.curvature_matrix:
            case 'hessian':
                step_size_kwargs['value_and_grad_fn'] = self.value_and_grad_fn
                step_size_fn = learning_rate_from_hessian
            case 'fisher':
                step_size_kwargs['model_fn'] = self.model_fn
                step_size_kwargs['loss_name'] = self.loss_name
                step_size_fn = learning_rate_from_fisher
            case _: raise ValueError(f"Unknown curvature matrix {self.curvature_matrix}")
        unclipped_step_size, step_size_second_order_term, grad_alignment = step_size_fn(**step_size_kwargs)
        new_optimiser_state['last_curvature'] = step_size_second_order_term

        step_size = unclipped_step_size
        if self.lr_clipping:
            step_size = jnp.clip(unclipped_step_size, a_min=-self.lr_clipping, a_max=self.lr_clipping)
        step_size = step_size * self.update_amplification
        if self.lr_ema:
            step_size = self.lr_ema*optimiser_state['lr_ema'] + (1-self.lr_ema)*step_size
            new_optimiser_state['lr_ema'] = step_size
        param_updates = jax.tree_util.tree_map(
            lambda dir, param: step_size*dir - self.weight_decay*param,
            direction,
            params)
        if self.update_norm_clipping:
            param_update_norm = jnp.linalg.norm(jax.flatten_util.ravel_pytree(param_updates)[0])
            param_updates = jax.lax.cond(param_update_norm > self.update_norm_clipping,
                                         lambda updates: jax.tree_util.tree_map(
                                             lambda upd: upd * self.update_norm_clipping / param_update_norm,
                                             updates),
                                         lambda updates: updates,
                                         param_updates)
        envelope_factor = self.envelope_schedule(global_step_int)
        new_params = jax.tree_util.tree_map(
            lambda p, upd: p + envelope_factor*upd,
            params,
            param_updates)
        statistics.update(dict(loss=loss,
                               learning_rate=-step_size,
                               lr_curvature=step_size_second_order_term,
                               grad_direction_product=grad_alignment,
                               gradient_norm=jnp.linalg.norm(jax.flatten_util.ravel_pytree(gradient)[0]),
                               update_norm=jnp.linalg.norm(jax.flatten_util.ravel_pytree(param_updates)[0]),
                               envelope_factor=envelope_factor))

        if self.damping_increase_factor:
            match self.damping_curvature:
                case 'step_size':
                    second_order_model_term = step_size_second_order_term
                case 'adam_no_sqrt' | 'adam_sqrt':
                    ravelled_direction = jax.flatten_util.ravel_pytree(direction)[0]
                    ravelled_adam_fisher_diag = jax.flatten_util.ravel_pytree(optimiser_state['base'].nu)[0]
                    if self.damping_curvature == 'adam_sqrt':
                        ravelled_adam_fisher_diag = jnp.sqrt(ravelled_adam_fisher_diag)
                    second_order_model_term = ravelled_direction.T @ (
                        (optimiser_state['damping'] + ravelled_adam_fisher_diag) * ravelled_direction)
                case _: raise ValueError(f"Unknown damping curvature {self.damping_curvature}")

            new_loss = self.value_and_grad_fn(new_params, func_state, sub_rng3, batch)[0][0]
            statistics['true_change'] = new_loss - loss

            match self.damping_algorithm:
                case 'kfac':
                    (new_optimiser_state['damping'],
                    statistics['rho'],
                    statistics['model_change']) = self.kfac_update_damping(
                        loss,
                        new_loss,
                        optimiser_state['damping'],
                        step_size,
                        jax.flatten_util.ravel_pytree(gradient)[0],
                        jax.flatten_util.ravel_pytree(direction)[0],
                        second_order_model_term)
                case 'decreasing_loss':
                    new_params, new_optimiser_state['damping'], new_optimiser_state['base'] = jax.lax.cond(
                        new_loss <= loss,
                        lambda: (new_params, self.damping_decrease_factor*optimiser_state['damping'], new_optimiser_state['base']),
                        lambda: (params, self.damping_increase_factor*optimiser_state['damping'], optimiser_state['base']))
                case _: raise ValueError(f"Unknown damping algorithm {self.damping_algorithm}")

            statistics['damping'] = new_optimiser_state['damping']

        return new_params, new_optimiser_state, new_func_state, statistics

    def kfac_update_damping(self,
                            last_loss,
                            new_loss,
                            old_damping,
                            step_size,
                            ravelled_gradient,
                            ravelled_direction,
                            second_order_model_term):
        model_change = (0.5 * step_size**2 * second_order_model_term
                        + step_size * jnp.dot(ravelled_gradient, ravelled_direction))
        rho = (new_loss - last_loss) / model_change
        damping = jnp.where(
            rho > 3/4, self.damping_decrease_factor * old_damping,
            jnp.where(
                rho < 1/4, old_damping * self.damping_increase_factor,
                old_damping))
        if any(self.damping_clipping):
            damping = jnp.clip(damping, *self.damping_clipping)
        return damping, rho, model_change


class SGDQLROptimiser(WrappedQuadraticLROptimiser):
    @property
    def base_optimiser(self):
        return optax.identity


class AdamQLROptimiser(WrappedQuadraticLROptimiser):
    def __init__(self, *args, b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0, mu_dtype=None, **kwargs):
        super().__init__(*args, **kwargs,
                         b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype)

    @property
    def base_optimiser(self):
        return optax.scale_by_adam


class KFACwithDynamicKroneckerCorrections():

    def __init__(self,
                 value_and_grad_fn,
                 model_fn,
                 loss_name,
                 correction_optimiser,
                 curvature_metric,
                 correction_type,
                 initial_learned_correction=None,
                 auto_lr=False,
                 **kwargs):
        del model_fn, loss_name
        self.value_and_grad_fn = value_and_grad_fn
        self.kfac_optimiser = kfac_jax.Optimizer(
            **kwargs,
            value_and_grad_func=value_and_grad_fn,
            value_func_has_aux=True,
            value_func_has_state=True,
            value_func_has_rng=False,
            multi_device=False)
        self.exact_curvature = kfac_jax.ImplicitExactCurvature(
            lambda *args, **kwargs: self.value_and_grad_fn(*args, **kwargs)[0])
        self.correction_value_and_grad = jax.value_and_grad(
            self.compute_correction_metric, argnums=(1,))

        self.correction_type = correction_type
        assert correction_type in ('implicit',
                                   'explicit',
                                   'explicit_cholesky',
                                   'explicit_override',
                                   'explicit_override_cholesky')
        self.curvature_metric = curvature_metric
        self.initial_learned_correction = initial_learned_correction
        self.auto_lr = auto_lr

        self.correction_optimiser = getattr(
            optax, correction_optimiser['name'])(
                **{k: v for k, v in correction_optimiser.items() if k != 'name'})


    def init(self, model_params, rng, sample_batch, func_state):
        rng, sub_rng= jax.random.split(rng)
        kfac_state = self.kfac_optimiser.init(model_params,
                                              sub_rng,
                                              sample_batch,
                                              func_state=func_state)
        # Overwrite all Kronecker blocks with our own additive correction field
        init_kwargs = dict(weight=lambda factor: factor.weight,
                           raw_value=lambda factor: factor.raw_value)
        match self.correction_type:
            case 'implicit':
                accumulator_class = lambda arg: arg
                init_kwargs = dict(arg=lambda factor: factor)
            case 'explicit':
                accumulator_class = CorrectedWeightedMovingAverage
                initialisation = self.initial_learned_correction
                init_kwargs.update(dict(
                    additive_correction=lambda factor: initialisation * identity_term(factor.raw_value)))
            case 'explicit_cholesky':
                accumulator_class = CholeskyCorrectedWeightedMovingAverage
                initialisation = self.initial_learned_correction
                if initialisation is not None:
                    initialisation = jnp.sqrt(initialisation)
                else:
                    initialisation = 1
                    # # Zeroed Cholesky factors are a local minimum
                    # initialisation = 1e-3
                    # init_kwargs['weight'] = lambda factor: factor.weight + 10
                init_kwargs.update(dict(
                    additive_correction=lambda factor: initialisation * identity_term(factor.raw_value)))
            case 'explicit_override':
                accumulator_class = OverriddenWeightedMovingAverage
                init_kwargs.update(dict(
                    learned_correction=lambda factor: self.initial_learned_correction * identity_term(factor.raw_value),
                    raw_value=lambda factor: jnp.full_like(factor.raw_value, jnp.nan)))
            case 'explicit_override_cholesky':
                accumulator_class = CholeskyOverriddenWeightedMovingAverage
                initialisation = self.initial_learned_correction
                if initialisation is not None:
                    initialisation = jnp.sqrt(initialisation)
                else:
                    initialisation = 1
                init_kwargs.update(dict(
                    learned_correction=lambda factor: initialisation * identity_term(factor.raw_value),
                    raw_value=lambda factor: jnp.full_like(factor.raw_value, jnp.nan)))

        for block_state in kfac_state.estimator_state.blocks_states:
            if hasattr(block_state, 'diagonal_factors'):
                identity_term = lambda raw_value: jnp.ones_like(raw_value)
                block_state.diagonal_factors = tuple(
                    accumulator_class(
                        **{key: value(factor) for key, value in init_kwargs.items()})
                    for factor in block_state.diagonal_factors)
            elif hasattr(block_state, 'inputs_factor') and hasattr(block_state, 'outputs_factor'):
                identity_term = lambda raw_value: jnp.eye(*raw_value.shape)
                block_state.inputs_factor = accumulator_class(
                    **{key: value(block_state.inputs_factor) for key, value in init_kwargs.items()})
                block_state.outputs_factor = accumulator_class(
                    **{key: value(block_state.outputs_factor) for key, value in init_kwargs.items()})
            else:
                raise NotImplementedError("Unknown block type")

        # We only care about the data shape here, so this works for both implicit and explicit corrections
        correction_state = self.correction_optimiser.init(kfac_state.estimator_state)
        return kfac_state, correction_state

    def step(self,
             model_params,
             optimiser_state,
             func_state,
             batch,
             rng,
             global_step_int):
        kfac_state, correction_state = optimiser_state

        (new_model_params,
         new_kfac_state,
         new_model_state,
         new_statistics) = self.kfac_optimiser.step(
             model_params,
             kfac_state,
             func_state=func_state,
             batch=batch,
             rng=rng,
             global_step_int=global_step_int)
        new_statistics.update(new_statistics.pop('aux', {}))

        new_optimiser_state, correction_stats = self._jittable_step(
            batch,
            correction_state,
            new_model_params,
            new_kfac_state,
            new_model_state)

        new_statistics.update(correction_stats)
        return (new_model_params,
                new_optimiser_state,
                new_model_state,
                new_statistics)

    @partial(jax.jit, static_argnums=(0,))
    @util.except_ray_jit_only_once
    def _jittable_step(self,
                       batch,
                       correction_state,
                       new_model_params,
                       new_kfac_state,
                       new_model_state):
        new_kfac_state = deepcopy(new_kfac_state)
        gradient = self.value_and_grad_fn(new_model_params,
                                          new_model_state,
                                          batch)[1]
        correction_metric, (correction_gradient,) = self.correction_value_and_grad(
            gradient,
            new_kfac_state.estimator_state,
            new_model_params,
            new_model_state,
            batch,
            damping=new_kfac_state.damping)
        (correction_updates,
         new_correction_state) = self.correction_optimiser.update(
             correction_gradient,
             correction_state,
             new_kfac_state.estimator_state)

        if self.correction_type == 'implicit':
            update_target = 'raw_value'
        elif self.correction_type in ('explicit', 'explicit_cholesky'):
            update_target = 'additive_correction'
        elif self.correction_type.startswith('explicit_override'):
            update_target = 'learned_correction'

        if self.auto_lr:
            step_size = correction_line_search_by_jvp(
                self.correction_value_and_grad,
                gradient,
                correction_gradient,
                new_kfac_state.estimator_state,
                new_model_params,
                new_model_state,
                batch,
                correction_updates,
                damping=new_kfac_state.damping)
            correction_updates = jax.tree_util.tree_map(
                lambda update: step_size * update,
                correction_updates)
        else:
            step_size = None

        for block_def, block_update in zip(new_kfac_state.estimator_state.blocks_states,
                                           correction_updates.blocks_states):
            if hasattr(block_def, 'diagonal_factors'):
                for diagonal_factor, subblock_update in zip(block_def.diagonal_factors,
                                                            block_update.diagonal_factors):
                    setattr(diagonal_factor, update_target,
                            getattr(diagonal_factor, update_target)
                            + getattr(subblock_update, update_target))
            else:
                setattr(block_def.inputs_factor, update_target,
                        getattr(block_def.inputs_factor, update_target)
                        + getattr(block_update.inputs_factor, update_target))
                setattr(block_def.outputs_factor, update_target,
                        getattr(block_def.outputs_factor, update_target)
                        + getattr(block_update.outputs_factor, update_target))
        return (new_kfac_state,
                new_correction_state), {'correction_loss': correction_metric,
                                        'correction_step_size': step_size}

    def compute_correction_metric(self,
                                  gradient,
                                  kfac_estimator_state,
                                  model_params,
                                  model_state,
                                  batch,
                                  damping):
        if self.curvature_metric == 'approx_true_grad':
            fisher_gradient = self.exact_curvature.multiply_fisher(
                (model_params, model_state, batch),
                gradient)
            # Add contribution from damping: (F + lambda*I)g = Fg + lambda*g
            fisher_gradient = jax.tree_util.tree_map(
                lambda x, g: x + damping*g,
                fisher_gradient, gradient)
            gradient_comparator = self.kfac_optimiser._estimator.multiply_matpower(
                kfac_estimator_state,
                fisher_gradient,
                identity_weight=damping,
                power=-1,
                exact_power=False,
                use_cached=False,
                pmap_axis_name=None)
            metric_vector = jax.tree_util.tree_map(
                lambda x, y: x - y,
                gradient_comparator,
                gradient)
        elif self.curvature_metric == 'true_approx_grad':
            inv_approx_curvature_gradient = self.kfac_optimiser._estimator.multiply_matpower(
                kfac_estimator_state,
                gradient,
                identity_weight=damping,
                power=-1,
                exact_power=False,
                use_cached=False,
                pmap_axis_name=None)
            gradient_comparator = self.exact_curvature.multiply_fisher(
                (model_params, model_state, batch),
                inv_approx_curvature_gradient)
            metric_vector = jax.tree_util.tree_map(
                lambda x, y, iacg: x + damping*iacg - y,
                gradient_comparator,
                gradient,
                inv_approx_curvature_gradient)
        elif self.curvature_metric == 'separate_products':
            true_product = jax.tree_util.tree_map(
                lambda x, g: x + damping*g,
                self.exact_curvature.multiply_fisher(
                    (model_params, model_state, batch),
                    gradient),
                gradient)
            approx_product = self.kfac_optimiser._estimator.multiply(
                kfac_estimator_state,
                gradient,
                identity_weight=damping,
                exact_power=True,
                use_cached=False,
                pmap_axis_name=None)
            metric_vector = jax.tree_util.tree_map(
                lambda x, y: x - y,
                true_product,
                approx_product)
        else:
            raise TypeError(f'Unrecognised curvature_metric {self.curvature_metric}')
        return jnp.linalg.norm(
            jax.flatten_util.ravel_pytree(metric_vector)[0],
            ord=2)


@kfac_jax.utils.pytree_dataclass
class CorrectedWeightedMovingAverage(kfac_jax.utils.WeightedMovingAverage):
    additive_correction: kfac_jax.utils.TPyTree

    def __init__(self, weight, raw_value, additive_correction=None, **kwargs) -> None:
        super().__init__(weight, raw_value, **kwargs)
        self.additive_correction = additive_correction
        if additive_correction is None:
            self.additive_correction = jnp.zeros_like(self.raw_value)
        elif isinstance(additive_correction, float):
            self.additive_correction = additive_correction * jnp.eye(*raw_value.shape)

    @property
    def value(self):
        """The value of the underlying array's data structure."""
        return jax.tree_util.tree_map(
            # lambda x: jnp.tril(self.additive_correction) @ jnp.tril(self.additive_correction).T + (x / self.weight),
            lambda x: self.additive_correction + (x / self.weight),
            self.raw_value)


@kfac_jax.utils.pytree_dataclass
class CholeskyCorrectedWeightedMovingAverage(CorrectedWeightedMovingAverage):

    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    @property
    def value(self):
        """The value of the underlying array's data structure."""
        if self.additive_correction.squeeze().ndim == 1:
            uncholesky = lambda x: x * x
        else:
            uncholesky = lambda x: jnp.tril(x) @ jnp.tril(x).T
        return jax.tree_util.tree_map(
            lambda x: uncholesky(self.additive_correction) + (x / self.weight),
            self.raw_value)


@kfac_jax.utils.pytree_dataclass
class OverriddenWeightedMovingAverage(kfac_jax.utils.WeightedMovingAverage):
    learned_correction: kfac_jax.utils.TPyTree

    def __init__(self, weight, raw_value, learned_correction=None, **kwargs) -> None:
        super().__init__(weight, raw_value, **kwargs)
        self.learned_correction = learned_correction
        if learned_correction is None:
            self.learned_correction = jnp.zeros_like(self.raw_value)
        elif isinstance(learned_correction, float):
            self.learned_correction = learned_correction * jnp.eye(*raw_value.shape)

    @property
    def value(self):
        """The value of the underlying array's data structure."""
        return jax.tree_util.tree_map(
            lambda _: self.learned_correction / self.weight,
            self.raw_value)


@kfac_jax.utils.pytree_dataclass
class CholeskyOverriddenWeightedMovingAverage(OverriddenWeightedMovingAverage):

    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    @property
    def value(self):
        """The value of the underlying array's data structure."""
        if self.learned_correction.squeeze().ndim == 1:
            uncholesky = lambda x: x * x
        else:
            uncholesky = lambda x: jnp.tril(x) @ jnp.tril(x).T
        return jax.tree_util.tree_map(
            lambda _: uncholesky(self.learned_correction) / self.weight,
            self.raw_value)


class BaydinSGD():

    def __init__(self,
                 value_and_grad_fn,
                 model_fn,
                 loss_name,
                 initial_learning_rate,
                 hypergradient_learning_rate):
        del model_fn, loss_name
        self.value_and_grad_fn = value_and_grad_fn
        self.initial_learning_rate = initial_learning_rate
        self.hypergradient_learning_rate = hypergradient_learning_rate

    def init(self, model_params, *args, **kwargs):
        del args, kwargs
        return dict(learning_rate=jnp.array(self.initial_learning_rate, dtype=jnp.float32),
                    last_gradient=jnp.zeros_like(jax.flatten_util.ravel_pytree(model_params)[0]))

    @partial(jax.jit, static_argnums=(0,))
    @util.except_ray_jit_only_once
    def step(self,
             params,
             optimiser_state,
             batch,
             func_state,
             rng,
             global_step_int,
             *args,
             **kwargs):
        del args, kwargs, global_step_int

        ((loss, (new_func_state, statistics)),
         gradient) = self.value_and_grad_fn(params, func_state, rng, batch)
        ravelled_gradient = jax.flatten_util.ravel_pytree(gradient)[0]

        gradient_product = jnp.dot(ravelled_gradient, -optimiser_state['last_gradient'])
        learning_rate = optimiser_state['learning_rate'] - self.hypergradient_learning_rate * gradient_product

        new_params = jax.tree_util.tree_map(
            lambda p, grad: p - learning_rate * grad,
            params,
            gradient)

        new_optimiser_state=dict(learning_rate=learning_rate,
                                 last_gradient=ravelled_gradient)
        statistics.update(dict(loss=loss,
                               learning_rate=learning_rate))

        return new_params, new_optimiser_state, new_func_state, statistics
