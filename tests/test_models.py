import chex
import jax
import jax.numpy as jnp
import kfac_jax

import models


def test_model_initialisation(model_params_and_state):
    model_params, model_state = model_params_and_state
    assert all(
        jax.tree_util.tree_map(lambda x: (x != 0).all(),
                               model_params))
    # Model state may be correctly zeroed, so don't test
    assert all(
        jax.tree_util.tree_map(lambda x: jnp.isfinite(x).all(),
                               model_params))
    assert all(
        jax.tree_util.tree_map(lambda x: jnp.isfinite(x).all(),
                               model_state))


def test_model_output_and_labels_match(model,
                                       model_params_and_state,
                                       loss_fn,
                                       sample_batch,
                                       training_forward_pass_kwargs):
    sample_input, sample_output = sample_batch
    model_params, model_state = model_params_and_state
    model_output, _ = model.apply(model_params,
                                  model_state,
                                  sample_input,
                                  **training_forward_pass_kwargs)
    final_params_zeros = all((x == 0).all()
                             for x in jax.tree_util.tree_leaves(model_params)[-2:])
    assert ((not (model_output == 0).all()) or final_params_zeros)
    assert not (model_output == model_output[0]).all() or final_params_zeros
    if loss_fn.func == models.cross_entropy_loss:
        # Classification dataset;
        # sample_output is indices but model_output is probabilities
        assert model_output.shape[0] == sample_output.shape[0]
        assert model_output.shape[-1] == loss_fn.keywords['num_classes']
    else:
        # Regression dataset;
        # sample_output and model_output are both values
        chex.assert_equal_shape((sample_output, model_output))


def test_cross_entropy_loss(rng):
    logits = jax.random.normal(rng(), (100, 10))
    labels = jax.random.randint(rng(), (100,), 0, 9)
    trial_value = jnp.mean(
        jnp.take_along_axis(
            -jax.nn.log_softmax(logits),
            labels[:, None],
            axis=1))
    assert jnp.allclose(trial_value,
                        models.cross_entropy_loss(logits,
                                                  labels,
                                                  kfac_mask=jnp.ones_like(labels),
                                                  num_classes=10))
    logits = logits.at[-10:].set(jnp.nan)
    labels = labels.at[-10:].set(jnp.nan)
    padded_trial_value = jnp.nanmean(
        jnp.take_along_axis(
            -jax.nn.log_softmax(logits),
            labels[:, None],
            axis=1))
    kfac_mask = ~jnp.isnan(logits).any(axis=1)
    assert jnp.allclose(padded_trial_value,
                        models.cross_entropy_loss(logits,
                                                  labels,
                                                  kfac_mask=kfac_mask,
                                                  num_classes=10))
    kfac_loss = kfac_jax.CategoricalLogitsNegativeLogProbLoss(
        jnp.where(jnp.isfinite(logits), logits, 0),
        labels,
        kfac_mask,
        weight=kfac_mask.shape[0]/kfac_mask.sum()
    ).evaluate(labels).mean()
    assert jnp.allclose(padded_trial_value, kfac_loss)


def test_mse_loss(rng):
    predictions = jax.random.normal(rng(), (100, 1))
    targets = jax.random.normal(rng(), (100, 1))
    trial_value = 0.5 * jnp.mean((predictions - targets)**2)
    assert jnp.allclose(trial_value,
                        models.mse_loss(predictions,
                                        targets,
                                        kfac_mask=jnp.ones_like(targets)))
    predictions = predictions.at[-10:].set(jnp.nan)
    targets = targets.at[-10:].set(jnp.nan)
    padded_trial_value = jnp.nanmean(0.5 * (predictions - targets)**2)
    kfac_mask = ~jnp.isnan(predictions)
    assert jnp.allclose(padded_trial_value,
                        models.mse_loss(predictions,
                                        targets,
                                        kfac_mask=kfac_mask))
    # kfac_loss_obj = kfac_jax.NormalMeanNegativeLogProbLoss(
    #     jnp.where(jnp.isfinite(predictions), predictions, 0),
    #     targets
    # ).evaluate()
    # kfac_loss = jnp.nanmean(0.5*(kfac_loss_obj - 0.5*jnp.log(jnp.pi)))
    correction_weight = kfac_mask.shape[0] / kfac_mask.sum()
    kfac_loss = kfac_jax.NormalMeanNegativeLogProbLoss(
        jnp.where(jnp.isfinite(predictions), predictions, 0),
        jnp.where(jnp.isfinite(targets), targets, 0),
        weight=0.5*correction_weight
    ).evaluate().mean()
    kfac_loss = jnp.mean(kfac_loss) - correction_weight*0.25*jnp.log(jnp.pi)
    assert jnp.allclose(padded_trial_value, kfac_loss)
