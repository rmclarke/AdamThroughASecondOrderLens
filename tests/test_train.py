import os
import pickle

import chex
import kfac_jax
import jax
import jax.numpy as jnp
import optax
import pytest

import datasets
import extern
import optimisers
import train
import util


def test_all_params_updated(model_params_and_state,
                            one_optimiser_step):
    model_params, model_state = model_params_and_state
    new_model_params, _, new_model_state, _ = one_optimiser_step
    assert all(jax.tree_util.tree_map(lambda x, y: (x != y).all(),
                                      model_params, new_model_params))
    assert all(jax.tree_util.tree_map(lambda x, y: (x != y).all(),
                                      model_state, new_model_state))


def test_output_depends_on_all_inputs(model_params_and_state,
                                      sample_batch,
                                      forward_pass_fn):
    if jnp.issubdtype(sample_batch[1].dtype, jnp.integer):
        argnums = (0,)
    else:
        argnums = (0, 1)

    gradient_fn = jax.grad(
        lambda batch_input, batch_output: forward_pass_fn(
            *model_params_and_state,
            (batch_input, batch_output))[0],
        argnums=argnums)
    # Ensure the only zero gradients are associated with NaN padded inputs
    assert all(
        jax.tree_util.tree_map(lambda x: ((x != 0) + ~jnp.isfinite(sample_batch[0]).all(axis=1, keepdims=True)).all(),
                               gradient_fn(*sample_batch)))


def test_training_state_construction(training_state, model, optimiser):
    assert training_state.model is model
    assert training_state.optimiser is optimiser
    assert training_state.model_params is None
    assert training_state.model_state is None
    assert training_state.optimiser_state is None
    assert training_state.statistics is None
    assert training_state.global_step == 0


@pytest.mark.parametrize('rng', [lambda: jax.random.PRNGKey(202302071133)])
def test_training_state_init_and_advance(training_state,
                                         rng,
                                         sample_batch,
                                         training_forward_pass_kwargs,
                                         model_params_and_state,
                                         optimiser_state,
                                         one_optimiser_step,
                                         monkeypatch):
    # model_params_and_state is computed without jax.random.split'ing rng, but
    # the training_state does split. Can't monkeypatch jax.random.split() to
    # solve this, as it's presumably used inside the model initialiser. So
    # use the fixed rng here, and let the fixtures for model_params_and_state
    # and optimiser_state copy the split used in TrainingState.init().
    chex.register_dataclass_type_with_jax_tree_util(kfac_jax.Optimizer.State)

    training_state.init(rng(),
                        *sample_batch,
                        **training_forward_pass_kwargs)
    chex.assert_trees_all_close(training_state.model_params,
                                model_params_and_state[0])
    chex.assert_trees_all_close(training_state.model_state,
                                model_params_and_state[1])
    try:
        chex.assert_trees_all_close(training_state.optimiser_state,
                                    optimiser_state)
    except TypeError:
        # KFAC-JAX optimiser states don't play well as PyTrees, so try a special case
        if isinstance(optimiser_state, tuple):
            assert isinstance(optimiser_state[0], kfac_jax.Optimizer.State)
        else:
            assert isinstance(optimiser_state, kfac_jax.Optimizer.State)
        chex.assert_trees_all_close(
            jax.tree_util.tree_leaves(training_state.optimiser_state),
            jax.tree_util.tree_leaves(optimiser_state))

    training_state.advance(sample_batch, rng())
    chex.assert_trees_all_close(training_state.model_params,
                                one_optimiser_step[0])
    chex.assert_trees_all_close(training_state.model_state,
                                one_optimiser_step[2])
    try:
        chex.assert_trees_all_close(training_state.optimiser_state,
                                    one_optimiser_step[1])
    except TypeError:
        # KFAC-JAX optimiser states don't play well as PyTrees, so try a special case
        if isinstance(optimiser_state, tuple):
            assert isinstance(optimiser_state[0], kfac_jax.Optimizer.State)
        else:
            assert isinstance(optimiser_state, kfac_jax.Optimizer.State)
        chex.assert_trees_all_close(
            jax.tree_util.tree_leaves(training_state.optimiser_state),
            jax.tree_util.tree_leaves(one_optimiser_step[1]))

    assert training_state.global_step == 1


@pytest.mark.parametrize('rng', [lambda: jax.random.PRNGKey(202302071630)])
def test_training_state_save(training_state,
                             rng,
                             sample_batch,
                             model_params_and_state,
                             optimiser_state,
                             training_forward_pass_kwargs,
                             tmp_path):
    training_state.init(rng(),
                        *sample_batch,
                        **training_forward_pass_kwargs)
    save_file = os.path.join(tmp_path, 'save_file')
    global_step = training_state.global_step
    statistics = training_state.statistics
    training_state.save(save_file)

    with open(save_file, 'rb') as saved_file:
        saved_state = pickle.load(saved_file)
    assert saved_state.model is None
    assert saved_state.optimiser is None
    chex.assert_trees_all_close(saved_state.model_params,
                                model_params_and_state[0])
    chex.assert_trees_all_close(saved_state.model_state,
                                model_params_and_state[1])
    try:
        chex.assert_trees_all_close(saved_state.optimiser_state,
                                    optimiser_state)
    except TypeError:
        if isinstance(optimiser_state, tuple):
            assert isinstance(optimiser_state[0], kfac_jax.Optimizer.State)
        else:
            assert isinstance(optimiser_state, kfac_jax.Optimizer.State)
        chex.assert_trees_all_close(
            jax.tree_util.tree_leaves(training_state.optimiser_state),
            jax.tree_util.tree_leaves(optimiser_state))

    assert saved_state.global_step == global_step
    assert saved_state.statistics == statistics
    # Need to remove file otherwise space will fill up
    os.remove(save_file)


@pytest.mark.parametrize('is_training', [True, False])
@pytest.mark.parametrize('extra_kwargs_spec', util.all_combinations(['is_training']))
def test_construct_forward_pass_extra_kwargs(extra_kwargs_spec, is_training):
    extra_kwargs = train.construct_forward_pass_extra_kwargs(
        extra_kwargs_spec, is_training)
    if 'is_training' in extra_kwargs_spec:
        assert extra_kwargs.get('is_training') == is_training
    else:
        assert 'is_training' not in extra_kwargs_spec


def test_initialise_randomness(rng):
    initial_seed = rng()
    assert (train.initialise_randomness(initial_seed[0])
            != jax.random.split(initial_seed)).all()
    assert (train.initialise_randomness(initial_seed[1])
            != jax.random.split(initial_seed)).all()
    blank_seed = train.initialise_randomness(None)
    assert (blank_seed != 0).all()
    assert (blank_seed != jax.random.split(jax.random.PRNGKey(0))).all()


def test_create_optimiser(optimiser,
                          forward_pass_fn,
                          model_params_and_state,
                          sample_batch):
    if hasattr(optimisers, type(optimiser).__name__):
        chex.assert_trees_all_close(
            optimiser.value_and_grad_fn(*model_params_and_state, sample_batch),
            jax.value_and_grad(
                forward_pass_fn, has_aux=True)(
                    *model_params_and_state, sample_batch))
    else:
        chex.assert_trees_all_close(
            optimiser._value_and_grad_func(*model_params_and_state, sample_batch),
            jax.value_and_grad(
                forward_pass_fn, has_aux=True)(
                    *model_params_and_state, sample_batch))
        assert isinstance(optimiser, (extern.optax_wrapper.OptaxWrapper,
                                      kfac_jax.Optimizer))
        assert not getattr(optimiser, 'value_func_has_aux', False)
        assert getattr(optimiser, 'value_func_has_state', True)
        assert not getattr(optimiser, 'value_func_has_rng', False)
        if not isinstance(optimiser, kfac_jax.Optimizer):
            assert isinstance(optimiser._optax_optimizer,
                              optax.GradientTransformation)


def test_log_losses(model,
                    loss_fn,
                    sample_batch,
                    logger,
                    forward_pass_extra_kwargs,
                    one_optimiser_step,
                    training_state,
                    forward_pass_fn,
                    tmp_path,
                    dataset_config):
    new_model_params, new_optimiser_state, new_model_state, new_statistics = one_optimiser_step
    new_training_state = train.TrainingState(training_state.model,
                                             training_state.optimiser)
    new_training_state.model_params = new_model_params
    new_training_state.model_state = new_model_state
    new_training_state.optimiser_state = new_optimiser_state
    new_training_state.statistics = new_statistics
    new_training_state.global_step = 1

    train.log_losses(new_training_state,
                     model,
                     loss_fn,
                     sample_batch,
                     sample_batch,
                     logger,
                     forward_pass_extra_kwargs)
    logger.close()
    read_data = util.pandas_from_tensorboard(tmp_path)

    def get_tag(tag):
        return read_data[read_data['tag'] == tag].value.iloc[0]

    assert read_data['tag'].is_unique
    assert get_tag('Loss/Training') == new_statistics['loss']
    assert get_tag('Loss/Validation') == get_tag('Loss/Test')
    if 'is_training' not in forward_pass_extra_kwargs:
        updated_training_loss, _ = forward_pass_fn(new_model_params,
                                                   new_model_state,
                                                   sample_batch)
        assert jnp.allclose(get_tag('Loss/Validation'), updated_training_loss)
        assert jnp.allclose(get_tag('Loss/Test'), updated_training_loss)
    for tag in read_data['tag']:
        if tag.startswith('Loss/') or tag.startswith('Accuracy/'):
            continue
        assert tag.startswith('Adaptive/')
        value = get_tag(tag)
        # Convert TensorBoard tag to KFAC internal tag
        tag = tag[9:].lower()
        assert value in (getattr(new_optimiser_state, tag, None),
                         getattr(new_statistics, tag, None),
                         getattr(new_optimiser_state, 'get', lambda *_: None)(tag),
                         getattr(new_statistics, 'get', lambda *_: None)(tag))

    if getattr(datasets, dataset_config['name']).track_accuracies:
        assert get_tag('Accuracy/Training') == new_statistics['accuracy']
        assert 0 <= get_tag('Accuracy/Training') <= 1
        assert 0 <= get_tag('Accuracy/Validation') <= 1
        assert 0 <= get_tag('Accuracy/Test') <= 1
    else:
        assert new_statistics['accuracy'] == -1
        all_tags = set(read_data['tag'])
        for dataset in ('Training', 'Validation', 'Test'):
            assert f'Accuracy/{dataset}' not in all_tags


@pytest.mark.parametrize('rng', [lambda: jax.random.PRNGKey(202302091136)])
def test_train_loop(block_rng_splits,
                    model,
                    loss_fn,
                    optimiser,
                    split_datasets,
                    sample_batch,
                    batch_size,
                    rng,
                    logger,
                    forward_pass_extra_kwargs,
                    one_optimiser_step):
    (new_model_params,
     new_optimiser_state,
     new_model_state,
     _) = one_optimiser_step
    new_training_state = train.train(
        model,
        loss_fn,
        optimiser,
        (sample_batch, *split_datasets[1:]),  # Single batch of data for full epoch
        batch_size=batch_size,
        num_epochs=1,
        rng=rng(),
        logger=logger,
        model_kwarg_spec=forward_pass_extra_kwargs)
    chex.assert_trees_all_equal(new_training_state.model_params, new_model_params)
    chex.assert_trees_all_equal(new_training_state.model_state, new_model_state)
    try:
        chex.assert_trees_all_equal(new_training_state.optimiser_state, new_optimiser_state)
    except (TypeError, AssertionError):
        # KFAC-JAX optimiser states don't play well as PyTrees, so try a special case
        if isinstance(new_optimiser_state, tuple):
            assert isinstance(new_optimiser_state[0], kfac_jax.Optimizer.State)
            assert isinstance(new_training_state.optimiser_state[0], kfac_jax.Optimizer.State)
        else:
            assert isinstance(new_optimiser_state, kfac_jax.Optimizer.State)
            assert isinstance(new_training_state.optimiser_state, kfac_jax.Optimizer.State)
        chex.assert_trees_all_close(
            jax.tree_util.tree_leaves(new_training_state.optimiser_state),
            jax.tree_util.tree_leaves(new_optimiser_state))

    assert new_training_state.global_step == 1


def test_expected_data_cases(model_params_and_state,
                             sample_batch,
                             forward_pass_fn,
                             rng):
    baseline_batch = sample_batch
    baseline_output = forward_pass_fn(*model_params_and_state, baseline_batch)

    high_precision_batch = [tensor.astype(jnp.float64)
                            for tensor in sample_batch]
    high_precision_output = forward_pass_fn(*model_params_and_state,
                                            high_precision_batch)
    chex.assert_trees_all_close(baseline_output, high_precision_output)

    perturbed_batch = (
        sample_batch[0] * (0.000001*jax.random.normal(rng()) + 1),
        sample_batch[1])
    perturbed_output = forward_pass_fn(*model_params_and_state,
                                       perturbed_batch)
    num_params = jax.flatten_util.ravel_pytree(model_params_and_state[0])[0].shape[0]
    chex.assert_trees_all_close(baseline_output, perturbed_output,
                                rtol=1e-7*num_params)

    if jnp.issubdtype(sample_batch[1].dtype, jnp.floating):
        zero_batch = [jnp.zeros_like(tensor)
                      for tensor in sample_batch]
        zero_output = forward_pass_fn(*model_params_and_state,
                                      zero_batch)
        chex.assert_trees_all_close(zero_output[0], 0)
