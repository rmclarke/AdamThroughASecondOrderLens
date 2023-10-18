"""Definitions of models and loss functions."""

import jax
import jax.numpy as jnp
import optax
import haiku as hk
import kfac_jax
import transformers
from functools import partial

import util


def create_model(name, **constructor_kwargs):
    """Create and transform an instance of hk.nets.`name` using `kwargs`."""
    if name in globals():
        model_constructor = globals()[name]
    else:
        model_constructor = getattr(hk.nets, name)
    if 'activation' in constructor_kwargs:
        constructor_kwargs['activation'] = getattr(jax.nn, constructor_kwargs['activation'])
    return hk.transform_with_state(
        lambda x, **kwargs: model_constructor(**constructor_kwargs)(x, **kwargs))


def create_loss(name, **kwargs):
    """Create an instance of `name` using ``kwargs``."""
    loss_function = globals()[name]
    return partial(loss_function, **kwargs)


def cross_entropy_loss(logits, labels, kfac_mask, num_classes):
    """Cross-entropy loss function, with necessary registration calls for KFAC-JAX."""
    # KFAC_JAX needs to be told to ignore padded data, but `mask` will only zero it,
    # so also set a corresponding correcting `weight`
    kfac_jax.register_softmax_cross_entropy_loss(
        jnp.where(jnp.isfinite(logits), logits, 0),
        labels,
        mask=kfac_mask,
        weight=kfac_mask.shape[0]/kfac_mask.sum())
    one_hot_labels = jax.nn.one_hot(labels, num_classes=num_classes)
    return jnp.nanmean(
        optax.softmax_cross_entropy(logits, one_hot_labels))


def mse_loss(predictions, targets, kfac_mask):
    """MSE loss function, with necessary registration calls for KFAC-JAX."""
    # KFAC's MSE loss is really a Gaussian distribution, so extra factors of log(sqrt(pi))
    # are picked up on each term. We can't correct for that effect before averaging,
    # so ignore it. The result is that KFAC's internal loss is higher than our reported
    # loss by 0.125 * log(pi) * batch_size/(batch_size - num_non_finite_points)
    kfac_jax.register_squared_error_loss(
        jnp.where(jnp.isfinite(predictions), predictions, 0),
        jnp.where(jnp.isfinite(targets), targets, 0),
        weight=0.5*kfac_mask.shape[0]/kfac_mask.sum())
    return jnp.nanmean(
        optax.l2_loss(predictions, targets))


def null_loss(model_output, *args, **kwargs):
    """Identity loss function, returning the model output directly."""
    del args, kwargs
    kfac_jax.register_squared_error_loss(model_output, jnp.zeros(1))
    return model_output.squeeze()


class KroneckerFactoredQuadraticModel(hk.Module):

    def __init__(self, num_samples, dimension):
        super().__init__()
        self.dimension = dimension
        self.num_samples = num_samples

    def init_true_curvature(self, *_):
        left_vectors = jax.random.normal(hk.next_rng_key(), (self.num_samples, self.dimension))
        right_vectors = jax.random.normal(hk.next_rng_key(), (self.num_samples, self.dimension))

        left_factors = jnp.stack(
            [jnp.outer(vector, vector)
             for vector in left_vectors])
        right_factors = jnp.stack(
            [jnp.outer(vector, vector)
                for vector in right_vectors])

        full_matrices = jnp.stack(
            [jnp.kron(left, right)
                for left, right in zip(left_factors, right_factors)])
        return full_matrices.mean(axis=0)

    def evaluate(self, data, true_curvature, linear_coeff):
        return (0.5 * (data.T @ true_curvature @ data)
                + (data.T @ linear_coeff))

    def __call__(self, _):
        """Compute regret."""
        data = hk.get_parameter("data",
                                shape=(self.dimension**2,),
                                init=hk.initializers.RandomNormal())
        true_curvature = hk.get_state("true_curvature",
                                      shape=(self.dimension**2, self.dimension**2),
                                      init=self.init_true_curvature)
        linear_coeff = hk.get_state("linear_coeff",
                                    shape=(self.dimension**2,),
                                    init=hk.initializers.RandomNormal())
        min_value = hk.get_state(
            "min_value",
            shape=(),
            init=hk.initializers.Constant(
                self.evaluate(
                    jnp.linalg.solve(
                        true_curvature, -linear_coeff),
                    true_curvature,
                    linear_coeff)))
        return self.evaluate(data, true_curvature, linear_coeff) - min_value


class RosenbrockModel(hk.Module):

    def __init__(self, a, b, initial_position=None):
        super().__init__()
        self.a = a
        self.b = b
        if initial_position is None:
            self.initial_position = hk.initializers.RandomNormal()
        else:
            self.initial_position = hk.initializers.Constant(
                jnp.array(initial_position))

    def __call__(self, _):
        x, y = hk.get_parameter("position",
                                shape=(2,),
                                init=self.initial_position)
        return util.rosenbrock(x, y, self.a, self.b)


class StackedLSTM(hk.Module):
    def __init__(self,
                 num_layers,
                 hidden_units_per_layer,
                 num_tokens,
                 num_embedding_dimensions,
                 embedding_dropout,
                 input_dropout,
                 inter_layer_dropout,
                 output_dropout):
         super().__init__()
         # With reference to Salesforce AWD-LSTM:
         # dropouti = input_dropout
         # dropouth = inter_layer_dropout
         # dropout = output_dropout
         # dropoute = embedding_dropout
         assert num_layers >= 1
         self.num_layers = num_layers
         self.hidden_units_per_layer = hidden_units_per_layer
         self.num_tokens = num_tokens
         self.num_embedding_dimensions = num_embedding_dimensions

         self.embedding_dropout = embedding_dropout
         self.input_dropout = input_dropout
         self.inter_layer_dropout = inter_layer_dropout
         self.output_dropout = output_dropout

    def __call__(self, data, is_training):
        lstm_layers = [hk.LSTM(self.hidden_units_per_layer)
                       for _ in range(self.num_layers)]
        lstm_states = [
            hk.LSTMState(
                hidden=hk.get_state(f'lstm_state{core_id}/hidden',
                                    shape=(self.hidden_units_per_layer,),
                                    init=lambda *_: core.initial_state(data.shape[1]).hidden),
                cell=hk.get_state(f'lstm_state{core_id}/cell',
                                  shape=(self.hidden_units_per_layer,),
                                  init=lambda *_: core.initial_state(data.shape[1]).cell))
            for core_id, core in enumerate(lstm_layers)]

        embedding_matrix = hk.get_parameter('embedding_matrix',
                                        shape=(self.num_tokens,
                                               self.num_embedding_dimensions),
                                        init=partial(jax.random.uniform,
                                                     hk.next_rng_key(),
                                                     minval=-0.1,
                                                     maxval=0.1))
        embedding_matrix = self.embedding_matrix_dropout(embedding_matrix,
                                                  hk.next_rng_key(),
                                                  self.embedding_dropout,
                                                  is_training)
        embedded_data = hk.Embed(embedding_matrix=embedding_matrix)(data)

        input_data = self.locked_dropout(embedded_data,
                                         hk.next_rng_key(),
                                         self.input_dropout,
                                         is_training)
        for layer, (state_id, state) in zip(lstm_layers, enumerate(lstm_states)):
            hidden_data, state = hk.dynamic_unroll(layer, input_data, state)
            # NOTE: WeightDrop?
            if layer != lstm_layers[-1]:
                input_data = self.locked_dropout(hidden_data,
                                                 hk.next_rng_key(),
                                                 self.inter_layer_dropout,
                                                 is_training)
            hk.set_state(f'lstm_state{state_id}/hidden', state.hidden)
            hk.set_state(f'lstm_state{state_id}/cell', state.cell)

        output_data = self.locked_dropout(hidden_data,
                                          hk.next_rng_key(),
                                          self.output_dropout,
                                          is_training)
        decoded_data = hk.Linear(self.num_tokens)(output_data)
        return decoded_data

    def reshaped_dropout(self, x, rng, dropout_rate, is_training, mask_shape):
        # From Salesforce AWD-LSTM code
        if not is_training:
            return x
        mask = jax.random.bernoulli(rng,
                                    1 - dropout_rate,
                                    shape=mask_shape)
        rescaled_mask = mask / (1 - dropout_rate)
        return rescaled_mask * x

    def locked_dropout(self, x, rng, dropout_rate, is_training):
        if not dropout_rate:
            return x
        mask_shape = (1, *x.shape[1:])
        return self.reshaped_dropout(x, rng, dropout_rate, is_training, mask_shape)

    def embedding_matrix_dropout(self, x, rng, dropout_rate, is_training):
        if not dropout_rate:
            return x
        mask_shape = (x.shape[0], 1)
        return self.reshaped_dropout(x, rng, dropout_rate, is_training, mask_shape)


class FlaxGPT2(hk.Module):

    def __init__(self, pretrained):
        super().__init__()
        if pretrained:
            self.flax_model = transformers.FlaxGPT2LMHeadModel.from_pretrained('gpt2')
        else:
            model_config = transformers.GPT2Config.from_pretrained('gpt2')
            model_config.vocab_size = 10000
            model_config.bos_token_id = 24  # Not specified; pre-trained config copies '<eos>'
            model_config.eos_token_id = 24  # '<eos>'
            model_config.unk_token_id = 26  # '<unk>'
            self.flax_model = transformers.FlaxGPT2LMHeadModel(model_config)
        self.initial_params = util.nested_to_flat_dict(self.flax_model.params)

    def __call__(self, data):
        transformer_params = util.flat_to_nested_dict(
            {key: hk.get_parameter(key,
                                   shape=value.shape,
                                   dtype=value.dtype,
                                   init=lambda *_: value)
             for key, value in self.initial_params.items()})
        output = self.flax_model(data, params=transformer_params)
        return output.logits
