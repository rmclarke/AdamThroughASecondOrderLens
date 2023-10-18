from copy import deepcopy

import chex
import jax
import jax.numpy as jnp
import kfac_jax
import optax
import pytest

import datasets
import models
import optimisers
import train
from extern.optax_wrapper import OptaxWrapper


def test_one_gradient_step_decreases_loss(forward_pass_fn,
                                          model_params_and_state,
                                          sample_batch,
                                          one_optimiser_step,
                                          optimiser_state,
                                          optimiser_config):
    initial_optimiser_state = deepcopy(optimiser_state)
    initial_model_params, initial_model_state = model_params_and_state
    initial_loss, _ = forward_pass_fn(initial_model_params, initial_model_state, sample_batch)
    new_model_params, new_optimiser_state, new_model_state, _ = one_optimiser_step
    final_loss, _ = forward_pass_fn(new_model_params, new_model_state, sample_batch)
    assert initial_loss != 0
    assert final_loss < initial_loss

    exempt_leaves = []
    if optimiser_config['optimiser']['name'] == 'KFACwithDynamicKroneckerCorrections':
        # Correction optimiser state contains unupdated cache quantities for
        # cleanliness of code, so allow these to be unchanged
        for block_state in initial_optimiser_state[1][0].mu.blocks_states:
            if block_state.cache is not None:
                exempt_leaves.extend(block_state.cache['-1'].values())
            if optimiser_config['optimiser']['correction_type'] in ('explicit_override', 'explicit_override_cholesky'):
                if hasattr(block_state, 'diagonal_factors'):
                    for diagonal_factor in block_state.diagonal_factors:
                        exempt_leaves.append(diagonal_factor.raw_value)
                else:
                    exempt_leaves.append(block_state.inputs_factor.raw_value)
                    exempt_leaves.append(block_state.outputs_factor.raw_value)
        for block_state in initial_optimiser_state[1][0].nu.blocks_states:
            if block_state.cache is not None:
                exempt_leaves.extend(block_state.cache['-1'].values())
            if optimiser_config['optimiser']['correction_type'] in ('explicit_override', 'explicit_override_cholesky'):
                if hasattr(block_state, 'diagonal_factors'):
                    for diagonal_factor in block_state.diagonal_factors:
                        exempt_leaves.append(diagonal_factor.raw_value)
                else:
                    exempt_leaves.append(block_state.inputs_factor.raw_value)
                    exempt_leaves.append(block_state.outputs_factor.raw_value)
        exempt_leaves.append(initial_optimiser_state[0].damping)
    if optimiser_config['optimiser']['name'] == 'kfac_jax':
        exempt_leaves.append(initial_optimiser_state.damping)
    if isinstance(initial_optimiser_state, dict) and 'damping' in initial_optimiser_state:
        exempt_leaves.append(initial_optimiser_state['damping'])

    for old_leaf, new_leaf in zip(jax.tree_util.tree_leaves(initial_optimiser_state),
                                  jax.tree_util.tree_leaves(new_optimiser_state)):
        assert (not (old_leaf == new_leaf).all()
                or any(old_leaf is exempt_leaf for exempt_leaf in exempt_leaves)
                # # Large models may have such small values that they collapse to zero.
                # # Treat these with the benefit of the doubt
                # or (new_leaf.shape == () and new_leaf.item() == 0)
                )


@pytest.mark.parametrize('initial_damping,' 'damping_adjustment_factor,' 'damping_clipping',
                         [(0, None, (None, None)),
                          (1, 0.5, (0.75, 1.25))])
@pytest.mark.parametrize('base_optimiser',
                         ['SGDQLROptimiser', 'AdamQLROptimiser'])
def test_qlr_optimisers(forward_pass_fn,
                        model_params_and_state,
                        sample_batch,
                        rng,
                        base_optimiser,
                        initial_damping,
                        damping_adjustment_factor,
                        damping_clipping):
    value_and_grad_fn = jax.value_and_grad(forward_pass_fn, has_aux=True)
    model_params, model_state = model_params_and_state
    (initial_loss, _), gradient = value_and_grad_fn(model_params,
                                                    model_state,
                                                    sample_batch)

    adam_qlr = getattr(optimisers, base_optimiser)(
        value_and_grad_fn,
        initial_damping=initial_damping,
        damping_adjustment_factor=damping_adjustment_factor,
        damping_clipping=damping_clipping)
    adam = OptaxWrapper(
        optax_optimizer=getattr(optax, base_optimiser[:-12].lower())(0.001),
        value_and_grad_func=value_and_grad_fn,
        value_func_has_aux=False,
        value_func_has_state=True,
        value_func_has_rng=False)

    adam_qlr_state = adam_qlr.init(model_params)
    # Unused empty scale_by_learning_rate state
    adam_state = adam.init(model_params, rng(), sample_batch)
    chex.assert_trees_all_close(adam_qlr_state['base'], adam_state[0])

    (adam_qlr_params,
     adam_qlr_new_state,
     adam_qlr_model_state,
     adam_qlr_statistics) = adam_qlr.step(model_params,
                                          adam_qlr_state,
                                          sample_batch,
                                          model_state)
    (adam_params,
     adam_new_state,
     adam_model_state,
     adam_statistics) = adam.step(model_params,
                                  adam_state,
                                  func_state=model_state,
                                  batch=sample_batch,
                                  rng=rng(),
                                  global_step_int=0)
    chex.assert_trees_all_close(adam_qlr_new_state['base'], adam_new_state[0], atol=1e-8)
    chex.assert_trees_all_close(adam_qlr_model_state, adam_model_state)
    adam_qlr_update = jax.tree_util.tree_map(lambda qlr, model: qlr - model,
                                             adam_qlr_params,
                                             model_params)
    adam_update = jax.tree_util.tree_map(lambda adam, model: adam - model,
                                         adam_params,
                                         model_params)
    ravelled_adam_qlr_update, _ = jax.flatten_util.ravel_pytree(adam_qlr_update)
    ravelled_adam_update, _ = jax.flatten_util.ravel_pytree(adam_update)
    assert jnp.allclose(ravelled_adam_qlr_update / jnp.linalg.norm(ravelled_adam_qlr_update),
                        ravelled_adam_update / jnp.linalg.norm(ravelled_adam_update),
                        atol=3.5e-5)
    assert jnp.allclose(adam_statistics['loss'], initial_loss)
    assert jnp.allclose(adam_qlr_statistics['loss'], initial_loss)
    assert jnp.allclose(adam_qlr_statistics['loss'], adam_statistics['loss'])

    if len(ravelled_adam_update) > 22360:
        # Hessian will be too large (> c. 2GB); don't try to compare
        return
    ravelled_params, unraveller = jax.flatten_util.ravel_pytree(model_params)
    ravelled_gradient, _ = jax.flatten_util.ravel_pytree(gradient)
    ravelled_direction, _ = jax.flatten_util.ravel_pytree(adam_update)
    hessian = jax.hessian(
        lambda params: forward_pass_fn(
            unraveller(params),
            model_state,
            sample_batch)[0])
    hessian_direction = hessian(ravelled_params) @ ravelled_direction
    step_size = -(jnp.dot(ravelled_gradient, ravelled_direction)
                  / jnp.dot(ravelled_direction, hessian_direction))
    if damping_adjustment_factor:
        new_damping = adam_qlr_new_state['damping']
        if adam_qlr_statistics['rho'] > 3/4:
            assert new_damping == jnp.clip(damping_adjustment_factor * initial_damping,
                                           *damping_clipping)
        elif adam_qlr_statistics['rho'] < 1/4:
            assert new_damping == jnp.clip(damping_adjustment_factor / initial_damping,
                                           *damping_clipping)
        else:
            assert new_damping == jnp.clip(initial_damping,
                                           *damping_clipping)
    else:
        chex.assert_trees_all_close(adam_qlr_params,
                                    jax.tree_util.tree_map(
                                        lambda p, dir: p + step_size*dir,
                                        model_params,
                                        adam_update),
                                    atol=1e-5)

    chex.clear_trace_counter()
    adam_qlr = getattr(optimisers, base_optimiser)(
        value_and_grad_fn,
        initial_damping=initial_damping,
        damping_adjustment_factor=damping_adjustment_factor,
        damping_clipping=damping_clipping,
        update_amplification=2)
    adam_qlr_amplified_params, *_ = adam_qlr.step(model_params,
                                                  adam_qlr_state,
                                                  sample_batch,
                                                  model_state)
    for original, unamplified, amplified in zip(jax.tree_util.tree_leaves(model_params),
                                                jax.tree_util.tree_leaves(adam_qlr_params),
                                                jax.tree_util.tree_leaves(adam_qlr_amplified_params)):
        assert jnp.allclose(amplified - original, 2*(unamplified - original), atol=1e-7)
    assert adam_qlr_statistics['learning_rate'] > 0


@pytest.mark.parametrize('curvature_metric', ['approx_true_grad',
                                              'true_approx_grad',
                                              'separate_products'])
@pytest.mark.parametrize('correction_optimiser', [dict(name='sgd', learning_rate=0.001),
                                                  dict(name='adam', learning_rate=0.001)])
@pytest.mark.parametrize('correction_type, initial_learned_correction',
                         [('implicit', None),
                          ('explicit', None),
                          ('explicit_cholesky', None),
                          ('explicit', 1),
                          ('explicit_cholesky', 1),
                          ('explicit_override', 1),
                          ('explicit_override_cholesky', 1)])
@pytest.mark.parametrize('auto_lr', [True, False])
@pytest.mark.skip
def test_kfac_with_dynamic_kronecker_correction_optimiser(forward_pass_fn,
                                                          model_params_and_state,
                                                          curvature_metric,
                                                          sample_batch,
                                                          rng,
                                                          correction_optimiser,
                                                          correction_type,
                                                          initial_learned_correction,
                                                          auto_lr):
    value_and_grad_fn = jax.value_and_grad(forward_pass_fn, has_aux=True)
    model_params, model_state = model_params_and_state
    (initial_loss, _), gradient = value_and_grad_fn(model_params,
                                                    model_state,
                                                    sample_batch)
    optimiser = optimisers.KFACwithDynamicKroneckerCorrections(
        value_and_grad_fn,
        correction_optimiser=correction_optimiser,
        curvature_metric=curvature_metric,
        l2_reg=0,
        use_adaptive_learning_rate=True,
        use_adaptive_momentum=True,
        use_adaptive_damping=True,
        damping_adaptation_interval=1,
        inverse_update_period=1,
        initial_damping=1.0,
        auto_lr=auto_lr,
        correction_type=correction_type,
        initial_learned_correction=initial_learned_correction)
    kfac_state, correction_state = optimiser.init(model_params, rng(), sample_batch, model_state)

    (kfac_updated_model_params,
     kfac_updated_kfac_state,
     kfac_updated_model_state,
     _) = optimiser.kfac_optimiser.step(
         deepcopy(model_params),
         deepcopy(kfac_state),
         func_state=deepcopy(model_state),
         batch=sample_batch,
         rng=rng(),
         global_step_int=0)
    (kfac_updated_loss, _), kfac_updated_gradient = value_and_grad_fn(
        kfac_updated_model_params,
        kfac_updated_model_state,
        sample_batch)

    ravelled_gradient, _ = jax.flatten_util.ravel_pytree(gradient)
    kfac_updated_ravelled_gradient, _ = jax.flatten_util.ravel_pytree(kfac_updated_gradient)
    assert not jnp.allclose(ravelled_gradient, 0)
    assert not jnp.allclose(ravelled_gradient, ravelled_gradient.flatten()[0])
    assert not jnp.allclose(kfac_updated_ravelled_gradient, 0)
    assert not jnp.allclose(kfac_updated_ravelled_gradient, kfac_updated_ravelled_gradient.flatten()[0])
    assert not jnp.allclose(ravelled_gradient, kfac_updated_ravelled_gradient)
    assert not jnp.allclose(initial_loss, 0)
    assert not jnp.allclose(kfac_updated_loss, 0)
    assert not jnp.allclose(initial_loss, kfac_updated_loss)
    assert jnp.isfinite(initial_loss)
    assert jnp.isfinite(kfac_updated_loss)
    assert jnp.isfinite(ravelled_gradient).all()
    assert jnp.isfinite(kfac_updated_ravelled_gradient).all()

    match correction_type:
        case 'implicit':
            correction_key = 'raw_value'
            block_type = kfac_jax.utils.WeightedMovingAverage
        case 'explicit':
            correction_key = 'additive_correction'
            block_type = optimisers.CorrectedWeightedMovingAverage
        case 'explicit_cholesky':
            correction_key = 'additive_correction'
            block_type = optimisers.CholeskyCorrectedWeightedMovingAverage
        case 'explicit_override':
            correction_key = 'learned_correction'
            block_type = optimisers.OverriddenWeightedMovingAverage
        case 'explicit_override_cholesky':
            correction_key = 'learned_correction'
            block_type = optimisers.CholeskyOverriddenWeightedMovingAverage
        case _: raise
    corrections = []
    for block_state in kfac_state.estimator_state.blocks_states:
        assert type(block_state.inputs_factor) == block_type
        assert type(block_state.outputs_factor) == block_type
        corrections.append(
            getattr(block_state.inputs_factor, correction_key, None))
        corrections.append(
            getattr(block_state.outputs_factor, correction_key, None))

    kfac_state_comparison = map(jnp.allclose,
                                jax.tree_util.tree_leaves(kfac_updated_kfac_state),
                                jax.tree_util.tree_leaves(kfac_state))
    for index, value in enumerate(kfac_state_comparison):
        assert (
            (not value.all())
            or any(jax.tree_util.tree_leaves(kfac_state)[index] is ac
                   for ac in corrections)
            or jax.tree_util.tree_leaves(kfac_state)[index] is kfac_state.damping)

    assert not any(map(jnp.allclose,
                       jax.tree_util.tree_leaves(kfac_updated_model_params),
                       jax.tree_util.tree_leaves(model_params)))
    assert not any(map(jnp.allclose,
                       jax.tree_util.tree_leaves(kfac_updated_model_state),
                       jax.tree_util.tree_leaves(model_state)))
    assert all(map(lambda x: jnp.isfinite(x).all(), jax.tree_util.tree_leaves(kfac_updated_model_params)))
    assert all(map(lambda x: jnp.isfinite(x).all(), jax.tree_util.tree_leaves(kfac_updated_model_state)))
    if correction_type in ('explicit_override', 'explicit_override_cholesky'):
        assert all(map(lambda x, y: (jnp.isfinite(x).all() or (jnp.isfinite(x) == jnp.isfinite(y)).all()),
                       jax.tree_util.tree_leaves(kfac_updated_kfac_state),
                       jax.tree_util.tree_leaves(kfac_state)))
    else:
        assert all(map(lambda x: jnp.isfinite(x).all(), jax.tree_util.tree_leaves(kfac_updated_kfac_state)))

    correction_gradient = optimiser.correction_value_and_grad(
        kfac_updated_gradient,
        kfac_updated_kfac_state.estimator_state,
        kfac_updated_model_params,
        kfac_updated_model_state,
        sample_batch,
        damping=1.0)[1][0]
    ravelled_correction_gradient, _ = jax.flatten_util.ravel_pytree(correction_gradient)
    assert not jnp.allclose(ravelled_correction_gradient, 0)
    assert not jnp.allclose(ravelled_correction_gradient, ravelled_correction_gradient.flatten()[0])
    chex.assert_trees_all_equal_shapes(
        jax.tree_util.tree_leaves(correction_gradient),
        jax.tree_util.tree_leaves(kfac_state.estimator_state))
    assert jnp.isfinite(ravelled_correction_gradient).all()

    (_, new_optimiser_state, _, _) = optimiser.step(deepcopy(model_params),
                                                    deepcopy((kfac_state, correction_state)),
                                                    deepcopy(model_state),
                                                    sample_batch,
                                                    rng(),
                                                    global_step_int=0)
    for old_block, new_block in zip(kfac_state.estimator_state.blocks_states,
                                    new_optimiser_state[0].estimator_state.blocks_states):
        for factor in ('inputs_factor', 'outputs_factor'):
            old_factor = getattr(old_block, factor)
            new_factor = getattr(new_block, factor)
            assert not (getattr(old_factor, correction_key) == getattr(new_factor, correction_key)).all()
            if correction_type != 'implicit':
                assert not (new_factor.value == new_factor.raw_value).all()
            if correction_type == 'explicit_cholesky':
                assert (new_factor.additive_correction == jnp.tril(new_factor.additive_correction)).all()
            if correction_type == 'explicit_override_cholesky':
                assert (new_factor.learned_correction == jnp.tril(new_factor.learned_correction)).all()


@pytest.mark.parametrize('damping', (0, 1))
def test_line_search_optimality(forward_pass_fn,
                                model_params_and_state,
                                sample_batch,
                                damping,
                                rng):
    # Strip state output from forward_pass_fn
    forward_pass = lambda *args, **kwargs: forward_pass_fn(*args, **kwargs)[0]
    grad_fn = jax.grad(forward_pass)
    model_params, model_state = model_params_and_state
    ravelled_params, param_unraveller = jax.flatten_util.ravel_pytree(model_params)

    gradient = grad_fn(model_params, model_state, sample_batch)
    ravelled_gradient, _ = jax.flatten_util.ravel_pytree(gradient)
    ravelled_direction = jax.random.normal(rng(), ravelled_params.shape)
    direction = param_unraveller(ravelled_direction)
    step_size = optimisers.line_search_by_jvp(
        jax.value_and_grad(forward_pass),
        model_params,
        model_state,
        gradient,
        direction,
        sample_batch,
        damping)

    flat_params, params_treedef = jax.tree_util.tree_flatten(model_params)

    def quadratic_model_change(step_size):
        param_change = step_size * ravelled_direction
        hessian_direction = jax.jvp(
            lambda *p: grad_fn(jax.tree_util.tree_unflatten(params_treedef, p),
                               model_state,
                               sample_batch),
            flat_params,
            jax.tree_util.tree_flatten(direction)[0])[1]
        ravelled_hessian_direction, _ = jax.flatten_util.ravel_pytree(hessian_direction)
        return (jnp.dot(param_change, ravelled_gradient)
                + 0.5*(step_size**2)*jnp.dot(ravelled_direction, ravelled_hessian_direction)
                + 0.5*(step_size**2)*damping*jnp.dot(ravelled_direction, ravelled_direction))

    assert jnp.allclose(jax.grad(quadratic_model_change)(step_size), 0, atol=1e-6)


@pytest.mark.parametrize('damping', (1,))
@pytest.mark.skip
def test_correction_line_search_optimality(forward_pass_fn,
                                           model_params_and_state,
                                           optimiser,
                                           one_optimiser_step,
                                           sample_batch,
                                           damping,
                                           rng):
    if not isinstance(optimiser, optimisers.KFACwithDynamicKroneckerCorrections):
        return
    forward_pass = lambda *args, **kwargs: forward_pass_fn(*args, **kwargs)[0]
    model_params, model_state = model_params_and_state
    gradient = jax.grad(forward_pass)(model_params, model_state, sample_batch)
    correction_grad_fn = jax.grad(optimiser.compute_correction_metric, argnums=(1,))

    correction_grad = correction_grad_fn(gradient,
                                         one_optimiser_step[1][0].estimator_state,
                                         model_params,
                                         model_state,
                                         sample_batch,
                                         damping=damping)[0]
    ravelled_correction_grad, correction_unraveller = jax.flatten_util.ravel_pytree(correction_grad)
    ravelled_direction = jax.random.normal(rng(), ravelled_correction_grad.shape)
    direction = correction_unraveller(ravelled_direction)
    step_size = optimisers.correction_line_search_by_jvp(
        jax.value_and_grad(optimiser.compute_correction_metric, argnums=(1,)),
        gradient,
        correction_grad,
        one_optimiser_step[1][0].estimator_state,
        model_params,
        model_state,
        sample_batch,
        direction,
        damping)

    (flat_correction_state,
     correction_treedef) = jax.tree_util.tree_flatten(
         one_optimiser_step[1][0].estimator_state)

    def quadratic_model_change(step_size):
        curvature_change = step_size * ravelled_direction
        hessian_direction = jax.jvp(
            lambda *cc: correction_grad_fn(
                gradient,
                jax.tree_util.tree_unflatten(correction_treedef, cc),
                model_params,
                model_state,
                sample_batch,
                damping),
            flat_correction_state,
            jax.tree_util.tree_flatten(direction)[0])[1]
        ravelled_hessian_direction, _ = jax.flatten_util.ravel_pytree(hessian_direction)
        return (jnp.dot(curvature_change, ravelled_correction_grad)
                + 0.5*(step_size**2)*jnp.dot(ravelled_direction, ravelled_hessian_direction)
                + 0.5*(step_size**2)*damping*jnp.dot(ravelled_direction, ravelled_direction))

    assert jnp.allclose(jax.grad(quadratic_model_change)(step_size), 0, atol=4e-5)


@pytest.mark.parametrize('curvature_metric',
                         ('approx_true_grad',
                          'separate_products',
                          'true_approx_grad',
                          ))
@pytest.mark.parametrize('rng', [lambda: jax.random.PRNGKey(202302091136)])
@pytest.mark.xfail
@pytest.mark.skip
# This test is too imprecise for assertions to validate properly
def test_explicit_corrections(curvature_metric, logger, rng):
    model = models.create_model(name='MLP', output_sizes=(1,), with_bias=True)
    loss_function = models.create_loss(name='mse_loss')
    split_datasets = datasets.make_split_datasets('UCI_Energy',
                                                  normalise_inputs=True,
                                                  normalise_outputs=True)
    def forward_pass(params, model_state, batch):
        return train.forward_pass(model=model,
                                  params=params,
                                  model_state=model_state,
                                  loss_function=loss_function,
                                  batch=batch)
    optimiser = train.create_optimiser(
        name='KFACwithDynamicKroneckerCorrections',
        forward_pass_fn=forward_pass,
        correction_optimiser=dict(name='adam', learning_rate=1e-0),
        auto_lr=True,
        curvature_metric=curvature_metric,
        correction_type='explicit_override',
        l2_reg=0,
        use_adaptive_learning_rate=True,
        use_adaptive_momentum=True,
        use_adaptive_damping=True,
        damping_adaptation_interval=1,
        inverse_update_period=1,
        initial_damping=1.0,
        initial_learned_correction=1e0)
    # Avoid logging and getting stuck on can_continue
    train.log_losses = lambda *_, **__: None

    def replacement_kfac_step(model_params,
                              kfac_state,
                              func_state,
                              batch,
                              *args, **kwargs):
        del args, kwargs
        kfac_state = deepcopy(kfac_state)
        statistics = dict(loss=forward_pass(model_params, func_state, batch)[0])
        for block_state in kfac_state.estimator_state.blocks_states:
            block_state.inputs_factor.weight = 1.0
            block_state.outputs_factor.weight = 1.0
        return model_params, kfac_state, func_state, statistics
    optimiser.kfac_optimiser.step = replacement_kfac_step

    final_state = train.train(
        rng=train.initialise_randomness(None),
        model=model,
        loss_function=loss_function,
        optimiser=optimiser,
        split_datasets=split_datasets,
        num_epochs=1000,
        batch_size=692,
        logger=logger,
        model_kwarg_spec=[])

    ravelled_new_params, param_unraveller = jax.flatten_util.ravel_pytree(final_state.model_params)
    jacobian = jax.jacrev(
        lambda p, *args: model.apply(param_unraveller(p), *args),
        has_aux=True)(
            ravelled_new_params,
            final_state.model_state,
            split_datasets[0][0].astype(jnp.float64)
        )[0].squeeze()
    # Batched outer product of vectors
    exact_fisher = jnp.einsum('bi,bj->ij', jacobian, jacobian) / 692
    optimiser_exact_fisher = jnp.stack(
        [jax.flatten_util.ravel_pytree(
            optimiser.exact_curvature.multiply_fisher(
                (final_state.model_params, final_state.model_state, split_datasets[0]),
                param_unraveller(jax.nn.one_hot(index, num_classes=ravelled_new_params.shape[0]))
            ))[0]
         for index in range(ravelled_new_params.shape[0])])
    assert jnp.allclose(exact_fisher, optimiser_exact_fisher)

    learned_fisher = jnp.stack(
        [jax.flatten_util.ravel_pytree(
            optimiser.kfac_optimiser.estimator.multiply(
                final_state.optimiser_state[0].estimator_state,
                param_unraveller(
                    jax.nn.one_hot(index,
                                   num_classes=ravelled_new_params.shape[0])
                ),
                identity_weight=0,
                exact_power=True,
                use_cached=False,
                pmap_axis_name=None)
            )[0]
         for index in range(ravelled_new_params.shape[0])])
    assert jnp.allclose(exact_fisher, learned_fisher)

    gradient = jax.grad(
        lambda *args: forward_pass(*args)[0]
    )(final_state.model_params,
      final_state.model_state,
      split_datasets[0])
    ravelled_gradient, _ = jax.flatten_util.ravel_pytree(gradient)
    damping = final_state.optimiser_state[0].damping
    exact_fisher_gradient = optimiser.exact_curvature.multiply_fisher(
        (final_state.model_params, final_state.model_state, split_datasets[0]),
        gradient)
    exact_fisher_gradient = jax.tree_util.tree_map(
        lambda x, g: x + damping*g,
        exact_fisher_gradient,
        gradient)
    inv_exact_fisher_gradient = (jnp.linalg.inv(exact_fisher) + damping*jnp.eye(*exact_fisher.shape)) @ ravelled_gradient
    approx_fisher_gradient = optimiser.kfac_optimiser.estimator.multiply(
        final_state.optimiser_state[0].estimator_state,
        gradient,
        identity_weight=damping,
        exact_power=False,
        use_cached=False,
        pmap_axis_name=None)
    inv_approx_fisher_gradient = optimiser.kfac_optimiser.estimator.multiply_matpower(
        final_state.optimiser_state[0].estimator_state,
        gradient,
        identity_weight=damping,
        power=-1,
        exact_power=False,
        use_cached=False,
        pmap_axis_name=None)
    inv_approx_fisher_gradient, _ = jax.flatten_util.ravel_pytree(inv_approx_fisher_gradient)
    chex.assert_trees_all_close(exact_fisher_gradient, approx_fisher_gradient)
    chex.assert_trees_all_close(inv_exact_fisher_gradient, inv_approx_fisher_gradient)
