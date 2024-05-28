"""Root script for starting individual training runs."""

import jax
import jax.numpy as jnp
import kfac_jax
import optax
from ray.air import session
from tensorboardX import SummaryWriter

import pickle
import json
import os
from functools import partial
from itertools import count

import config
import datasets
import extern.optax_wrapper
import models
import optimisers
import util
from datetime import datetime


class TrainingState():
    """Convenience wrapper around Haiku/KFAC-JAX state objects."""

    def __init__(self, model, optimiser, recurrent_state=False):
        """Construct blank state using objects provided."""
        self.model = model
        self.optimiser = optimiser
        self.model_params = None
        self.model_state = None
        self.optimiser_state = None
        self.statistics = None
        self.global_step = 0
        self.can_continue = True
        self.recurrent_state = recurrent_state

    def init(self, rng, sample_input, sample_output, **model_kwargs):
        """Initialise state to its full form."""
        rng1, rng2 = jax.random.split(rng)
        (self.model_params,
         self.model_state) = self.model.init(rng1,
                                             sample_input,
                                             **model_kwargs)
        self.optimiser_state = self.optimiser.init(self.model_params,
                                                   rng2,
                                                   (sample_input,
                                                    sample_output),
                                                   func_state=self.model_state)

    def advance(self, batch, rng):
        """Perform one optimiser step, advancing the state by one iteration."""
        (self.model_params,
         self.optimiser_state,
         self.model_state,
         self.statistics) = self.optimiser.step(
             self.model_params,
             self.optimiser_state,
             func_state=self.model_state,
             batch=batch,
             rng=rng,
             global_step_int=self.global_step)
        self.statistics.update(self.statistics.pop('aux', {}))
        self.global_step += 1

    def save(self, path):
        """Save the exportable components of `self` to `path`."""
        optimiser = self.optimiser
        model = self.model
        self.optimiser = None
        self.model = None
        with open(path, 'wb') as save_file:
            pickle.dump(self, save_file)
        self.optimiser = optimiser
        self.model = model

    # Must have current_model_state as an argument to avoid
    @partial(jax.jit, static_argnums=(0,), static_argnames=('is_training'))
    def maybe_fresh_model_state(self, current_model_state, rng, sample_input, **model_kwargs):
        """Provide a new model state if needed for an RNN."""
        if self.recurrent_state:
            return self.model.init(rng, sample_input, **model_kwargs)[1]
        else:
            return current_model_state


def construct_forward_pass_extra_kwargs(extra_kwargs, is_training=None):
    """Given the list of `extra_kwargs` required for this model, and
    corresponding specifiers of our current state, populate extra kwargs for
    the forward_pass call.
    """
    model_kwargs = {}
    if 'is_training' in extra_kwargs:
        model_kwargs['is_training'] = is_training
    return model_kwargs


def model_forward_pass(model,
                       params,
                       model_state,
                       rng,
                       batch,
                       **model_kwargs):
    """Compute a forward pass of `model` only, without calculating loss."""
    # Must pass finite data to the model, otherwise JAX grads break
    finite_data = jnp.isfinite(batch[0]).all(axis=range(1, batch[0].ndim), keepdims=True)
    model_output, state = model.apply(params,
                                      model_state,
                                      rng,
                                      jnp.where(finite_data, batch[0], 0),
                                      **model_kwargs)
    finite_data = finite_data.squeeze(range(batch[1].ndim+1, finite_data.ndim))
    if finite_data.ndim != model_output.ndim:
        # StackedLSTM breaks the above logic, so handle it separately
        finite_data = jnp.expand_dims(finite_data, range(finite_data.ndim, model_output.ndim))
    model_output = jnp.where(finite_data, model_output, jnp.nan)
    return model_output, (state, finite_data)


def full_forward_pass(model,
                      params,
                      model_state,
                      rng,
                      loss_function,
                      batch,
                      compute_accuracies=False,
                      **model_kwargs):
    """Compute a forward pass of `model` to compute a loss value."""
    model_output, (state, finite_data) = model_forward_pass(model, params, model_state, rng, batch, **model_kwargs)
    loss = loss_function(model_output, batch[1], kfac_mask=finite_data.squeeze())
    # For JITability, must always log a numerical accuracy
    statistics = dict(accuracy=-1)
    if compute_accuracies:
        statistics['accuracy']= util.top_1_accuracy(model_output, batch[1])
    return loss, (state, statistics)


non_training_forward_pass = jax.jit(full_forward_pass,
                                    static_argnames=('model',
                                                     'loss_function',
                                                     'compute_accuracies',
                                                     'is_training'))


def initialise_randomness(seed):
    """Reproducibly instantiate a random number generator."""
    if seed is None:
        # TODO: Check seeding is fair
        seed = int(datetime.now().timestamp() * 1e6)
    return jax.random.split(
        jax.random.PRNGKey(seed)
    )[0]


def create_optimiser(name, forward_pass_fn, model_forward_pass_fn, loss_name, total_steps, **kwargs):
    """Construct and wrap (if necessary) the optimiser `class_name`."""
    wrapper_kwargs = dict(
        value_and_grad_func=jax.value_and_grad(forward_pass_fn, has_aux=True),
        value_func_has_aux=True,
        value_func_has_state=True,
        value_func_has_rng=True)

    if kwargs.get('scaling_envelope_peak', None):
        kwargs['total_steps'] = total_steps

    if name == 'kfac_jax':
        if 'learning_rate' in kwargs:
            learning_rate = kwargs.pop('learning_rate')
            kwargs['learning_rate_schedule'] = lambda _: learning_rate
        if 'momentum' in kwargs:
            momentum = kwargs.pop('momentum')
            kwargs['momentum_schedule'] = lambda _: momentum
        if 'initial_damping' in kwargs and not kwargs['use_adaptive_damping']:
            initial_damping = kwargs.pop('initial_damping')
            kwargs['damping_schedule'] = lambda _: initial_damping
        optimiser = kfac_jax.Optimizer(**kwargs,
                                       **wrapper_kwargs,
                                       multi_device=False)
    elif hasattr(optax, name):
        weight_decay = kwargs.pop('add_decayed_weights', None)
        gradient_norm_clipping = kwargs.pop('clip_by_global_norm', None)
        if isinstance(kwargs.get('learning_rate', None), dict):
            scheduler_func = kwargs['learning_rate'].pop('name')
            kwargs['learning_rate'] = getattr(optax, scheduler_func)(**kwargs['learning_rate'])
        optimiser = optax.inject_hyperparams(getattr(optax, name))(**kwargs)
        if gradient_norm_clipping:
            optimiser = optax.chain(
                optax.clip_by_global_norm(gradient_norm_clipping),
                optimiser)
        if weight_decay:
            optimiser = optax.chain(
                optax.add_decayed_weights(weight_decay),
                optimiser)
        optimiser = extern.optax_wrapper.OptaxWrapper(
            optax_optimizer=optimiser, **wrapper_kwargs)
    elif hasattr(optimisers, name):
        optimiser = getattr(optimisers, name)(
            value_and_grad_fn=jax.value_and_grad(forward_pass_fn,
                                                 has_aux=True),
            model_fn=model_forward_pass_fn,
            loss_name=loss_name,
            **kwargs)
    else:
        raise NameError(f"Unknown optimiser {name}")

    return optimiser


def evaluate_output(state,
                    model,
                    loss_function,
                    dataset,
                    model_kwarg_spec,
                    rng,
                    track_accuracies):
    sub_rng1, sub_rng2 = jax.random.split(rng)
    extra_kwargs = construct_forward_pass_extra_kwargs(
        model_kwarg_spec,
        is_training=False)
    forward_pass_kwargs = dict(
        model=model,
        params=state.model_params,
        loss_function=loss_function,
        compute_accuracies=track_accuracies,
        **extra_kwargs)
    if state.recurrent_state:
        # Batch our evaluation passes, because the datasets are too big to evaluate in one go
        rng, sub_rng, sample_rng, cursor_rng = jax.random.split(sub_rng2, 4)
        cursor = jax.random.randint(cursor_rng, (1,), minval=0, maxval=len(dataset[0])-1400).item()
        sample_batch = (dataset[0][cursor:cursor+1400].reshape(20, 70).T,
                        dataset[1][cursor:cursor+1400].reshape(20, 70).T)
        # dataset = datasets.pad_dataset_for_equal_batches(dataset, 20)
        # sample_batch = next(datasets.make_batches(dataset, (70, 20), sample_rng, lambda _, data: data))
        model_state = state.maybe_fresh_model_state(state.model_state,
                                                    sub_rng1,
                                                    sample_batch[0],
                                                    **extra_kwargs)
        rng, sub_rng = jax.random.split(rng)
        return non_training_forward_pass(
            model_state=model_state,
            rng=sub_rng,
            batch=sample_batch,
            **forward_pass_kwargs)

        # total_loss = 0
        # for batch in datasets.make_batches(dataset, (1000, 20), sub_rng, lambda _, data: data):
        #     rng, sub_rng = jax.random.split(rng)
        #     loss, (model_state, _) = non_training_forward_pass(
        #         model_state=model_state,
        #         rng=sub_rng,
        #         batch=batch,
        #         **forward_pass_kwargs)
        #     loss = loss * jnp.prod(jnp.array(batch[0].shape))
        #     total_loss = total_loss + loss
        # total_loss = total_loss / dataset[0].shape[0]
        # return total_loss, (model_state, None)

    else:
        fresh_model_state = state.maybe_fresh_model_state(state.model_state,
                                                          sub_rng1,
                                                          dataset[0],
                                                          **extra_kwargs)
        return non_training_forward_pass(
            model_state=fresh_model_state,
            rng=sub_rng2,
            batch=dataset,
            **forward_pass_kwargs)


def log_losses(state,
               model,
               loss_function,
               validation_dataset,
               test_dataset,
               logger,
               model_kwarg_spec,
               rng):
    """Log training loss from `state`, then calculate and log other losses."""
    rng, *sub_rngs = jax.random.split(rng, 3)
    sub_rngs = iter(sub_rngs)
    del rng

    for key in ('rho',
                'damping',
                'learning_rate',
                'momentum',
                'model_change',
                'true_change',
                'lr_curvature',
                'grad_direction_product',
                'gradient_norm',
                'update_norm',
                'envelope_factor'):
        if (isinstance(state.optimiser_state, dict)
                and key in state.optimiser_state):
            value = state.optimiser_state[key]
        elif hasattr(state.optimiser_state, key):
            value = getattr(state.optimiser_state, key)
            if value is None:
                continue
        elif (hasattr(state.optimiser_state, 'hyperparams')
                and key in state.optimiser_state.hyperparams):
            value = state.optimiser_state.hyperparams[key]
        elif (isinstance(state.optimiser_state, tuple)
                and hasattr(state.optimiser_state[1], 'hyperparams')
                and key in state.optimiser_state[1].hyperparams):
            # Weight decay case
            value = state.optimiser_state[1].hyperparams[key]
        elif isinstance(state.statistics, dict) and key in state.statistics:
            value = state.statistics[key]
        elif hasattr(state.statistics, key):
            value = getattr(state.statistics, key)
        else:
            continue
        logger.add_scalar(f'Adaptive/{key.title()}',
                          value.item(),
                          state.global_step)

    for key in ('correction_loss', 'correction_step_size'):
        if key in state.statistics:
            logger.add_scalar(f'Correction/{key[11:].title()}',
                              state.statistics[key].item(),
                              state.global_step)

    rosenbrock_tag = None
    if 'rosenbrock_model' in state.model_params:
        rosenbrock_tag = 'rosenbrock_model'
    if 'rosenbrock_as_least_squares' in state.model_params:
        rosenbrock_tag = 'rosenbrock_as_least_squares'
    if rosenbrock_tag:
        position = state.model_params[rosenbrock_tag]['position']
        logger.add_scalar('Position/x',
                          position[0],
                          state.global_step)
        logger.add_scalar('Position/y',
                          position[1],
                          state.global_step)

    track_accuracies = bool(state.statistics['accuracy'] != -1)
    training_loss = state.statistics['loss'].item()
    if track_accuracies: training_accuracy = state.statistics['accuracy'].item()
    test_output = evaluate_output(state,
                                  model,
                                  loss_function,
                                  test_dataset,
                                  model_kwarg_spec,
                                  next(sub_rngs),
                                  track_accuracies)
    test_loss = test_output[0].item()
    if track_accuracies: test_accuracy = test_output[1][1]['accuracy'].item()

    if validation_dataset is not None:
        validation_output = evaluate_output(state,
                                            model,
                                            loss_function,
                                            validation_dataset,
                                            model_kwarg_spec,
                                            next(sub_rngs),
                                            track_accuracies)
        validation_loss = validation_output[0].item()
        logger.add_scalar('Loss/Validation',
                          validation_loss,
                          state.global_step)
        if track_accuracies:
            validation_accuracy = validation_output[1][1]['accuracy'].item()
            logger.add_scalar('Accuracy/Validation',
                            validation_accuracy,
                            state.global_step)
    else:
        validation_loss = None

    if not all(map(lambda x: x is None or jnp.isfinite(x),
                   (training_loss, validation_loss, test_loss))):
        state.can_continue = False

    if util.in_monitored_ray_session():
        session.report(dict(training_loss=training_loss,
                            validation_loss=validation_loss,
                            test_loss=test_loss))

    # Put at end to mitigate first-batch logging artifacts
    logger.add_scalar('Loss/Training',
                      training_loss,
                      state.global_step)
    logger.add_scalar('Loss/Test',
                      test_loss,
                      state.global_step)
    if track_accuracies:
        logger.add_scalar('Accuracy/Training',
                          training_accuracy,
                          state.global_step)
        logger.add_scalar('Accuracy/Test',
                          test_accuracy,
                          state.global_step)


def train(model,
          loss_function,
          optimiser,
          split_datasets,
          batch_size,
          num_epochs,
          rng,
          logger,
          model_kwarg_spec,
          recurrent_state,
          initial_state=None):
    """Core training loop."""
    (training_dataset,
     validation_dataset,
     test_dataset), input_augmenter = split_datasets
    rng, sub_rng = jax.random.split(rng)
    sample_batch = next(
        datasets.make_batches(
            training_dataset, batch_size, sub_rng, input_augmenter))
    training_extra_kwargs = construct_forward_pass_extra_kwargs(
        model_kwarg_spec,
        is_training=True)
    if initial_state:
        state = initial_state
    else:
        state = TrainingState(model, optimiser, recurrent_state)
        rng, sub_rng = jax.random.split(rng)
        state.init(sub_rng,
                   *sample_batch,
                   **training_extra_kwargs)

    for epoch in (range(num_epochs) if num_epochs else count()):
        rng, sub_rng, sub_rng2 = jax.random.split(rng, 3)
        for batch in datasets.make_batches(
                training_dataset, batch_size, sub_rng, input_augmenter):
            rng, sub_rng, sub_rng2, sub_rng3 = jax.random.split(rng, 4)
            state.model_state = state.maybe_fresh_model_state(
                state.model_state,
                sub_rng2,
                sample_batch[0],
                **training_extra_kwargs)
            state.advance(batch, rng=sub_rng)
            log_losses(state,
                       model,
                       loss_function,
                       validation_dataset,
                       test_dataset,
                       logger,
                       model_kwarg_spec,
                       sub_rng3)
            util.maybe_print('.', end='', flush=True)
            if not state.can_continue:
                return state
        if epoch % 1 == 0:
            training_loss = state.statistics['loss'].item()
            util.maybe_print(f'Epoch={epoch}  TrainingLoss={training_loss}', flush=True)

    return state


def main(config_dict=None, config_overrides={}):
    """Perform a training run."""
    if config_dict is None:
        config_dict = config.load_config(config_overrides)
    else:
        util.nested_update(config_dict, config_overrides)

    train_kwargs = dict(
        rng=initialise_randomness(config_dict.get('seed', None)),
        model=models.create_model(**config_dict['model']),
        loss_function=models.create_loss(**config_dict['loss']),
        split_datasets=datasets.make_split_datasets(
            pad_to_equal_training_batches=config_dict['batch_size'],
            **config_dict['dataset']),
        num_epochs=config_dict['num_epochs'],
        batch_size=config_dict['batch_size'],
        recurrent_state=config_dict.get('recurrent_model_state', None),
    )

    if config_dict['num_epochs'] == -1:
        train_kwargs['num_epochs'] = None
        total_steps = float('inf')
    else:
        total_steps = train_kwargs['num_epochs'] * len(train_kwargs['split_datasets'][0][0][0]) // jnp.prod(jnp.array(train_kwargs['batch_size']))

    if 'subsequence_length' in config_dict['dataset']:
        train_kwargs['batch_size'] = (config_dict['dataset']['subsequence_length'], config_dict['batch_size'])

    training_forward_pass_kwargs = construct_forward_pass_extra_kwargs(
        config_dict['forward_pass_extra_kwargs'],
        is_training=True)

    def wrapped_forward_pass_fn(params, model_state, rng, batch):
        return full_forward_pass(
            model=train_kwargs['model'],
            params=params,
            model_state=model_state,
            loss_function=train_kwargs['loss_function'],
            batch=batch,
            rng=rng,
            compute_accuracies=getattr(datasets, config_dict['dataset']['name']).track_accuracies,
            **training_forward_pass_kwargs)

    def wrapped_model_forward_pass_fn(params, model_state, rng, batch):
        return model_forward_pass(
            model=train_kwargs['model'],
            params=params,
            model_state=model_state,
            batch=batch,
            rng=rng,
            **training_forward_pass_kwargs)

    train_kwargs['optimiser'] = create_optimiser(
        forward_pass_fn=wrapped_forward_pass_fn,
        model_forward_pass_fn=wrapped_model_forward_pass_fn,
        loss_name=config_dict['loss']['name'],
        total_steps=total_steps,
        **config_dict['optimiser'])

    if config_dict.get('load_state', False):
        with open(config_dict['load_state'], 'rb') as state_file:
            # This state has no model or optimiser,
            # so calls to .init() will fail
            train_kwargs['initial_state'] = pickle.load(state_file)
    log_directory = config.log_directory(config_dict)
    with SummaryWriter(log_dir=log_directory) as logger:
        with open(os.path.join(log_directory,
                               'config.json'), 'w') as config_file:
            json.dump(config_dict, config_file)
        state = train(
            logger=logger,
            model_kwarg_spec=config_dict['forward_pass_extra_kwargs'],
            **train_kwargs)
    if config_dict.get('save_state', False):
        state.save(config_dict['save_state'])


if __name__ == '__main__':
    main()
