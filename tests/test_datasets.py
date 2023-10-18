from functools import partial

import chex
import numpy as np
import jax.numpy as jnp
import pytest

import datasets
import models


def test_no_leakage(split_datasets):
    if split_datasets[1] is None:
        # Ignore empty validation set
        split_datasets = [split_datasets[0], split_datasets[2]]
    # combined_input_output_datasets = map(lambda dataset: zip(*dataset),
    #                                      split_datasets)
    # dataset_sets = [set(map(tuple, dataset.tolist()))
    #                 for dataset in combined_input_output_datasets]
    dataset_sets = [
        set(
            (input_point.tobytes(), output_point.tobytes())
            for input_point, output_point in zip(*dataset))
        for dataset in split_datasets]
    for set_idx, set1 in enumerate(dataset_sets[:-1]):
        for set2 in dataset_sets[set_idx+1:]:
            assert set2.isdisjoint(set1)


def test_dataset_splitting(dataset_config,
                           split_datasets,
                           monkeypatch):
    if dataset_config['validation_proportion'] == 0:
        assert split_datasets[1] is None
        split_datasets = [split_datasets[0], split_datasets[2]]
    else:
        assert split_datasets[1] is not None
        # Training and validation sets are normalised as one, so must
        # combine to check the normalisation
        split_datasets = [(jnp.concatenate((split_datasets[0][0],
                                            split_datasets[1][0]), axis=0),
                           jnp.concatenate((split_datasets[0][1],
                                            split_datasets[1][1]), axis=0)),
                          split_datasets[2]]

    normalise_dimension = (0, 1) if split_datasets[0][0].ndim == 3 else 0
    for dataset_idx, (input_data,
                      output_data) in enumerate(split_datasets):
        if dataset_idx == 0:
            # Training/validation dataset; expect perfect normalisation
            close_atol = 1e-6
            close_rtol = 1e-5  # default
        else:
            # Test dataset; expect imperfect normalisation
            close_atol = 0.2
            close_rtol = max(input_data.shape[1] / 300, 0.6)
        if dataset_config['normalise_inputs']:
            assert jnp.allclose(
                jnp.nanmean(input_data, axis=normalise_dimension), 0,
                atol=close_atol)
            assert jnp.allclose(
                jnp.nanstd(input_data, axis=normalise_dimension), 1,
                rtol=close_rtol)
        if dataset_config['normalise_outputs']:
            assert jnp.allclose(
                jnp.nanmean(output_data, axis=normalise_dimension), 0,
                atol=close_atol)
            assert jnp.allclose(
                jnp.nanstd(output_data, axis=normalise_dimension), 1,
                rtol=close_rtol)


def test_batches(split_datasets, rng):
    batches = list(datasets.make_batches(split_datasets[0],
                                         128,
                                         rng()))
    chex.assert_equal_shape((split_datasets[0][0],
                             jnp.concatenate([batch[0] for batch in batches])))
    chex.assert_equal_shape((split_datasets[0][1],
                             jnp.concatenate([batch[1] for batch in batches])))

    combined_source_data = set(
        (input_data.tobytes(), output_data.tobytes())
        for input_data, output_data in zip(*split_datasets[0]))
    combined_batch_data = set(
        (input_data.tobytes(), output_data.tobytes())
        for batch in batches
        for input_data, output_data in zip(*batch))
    assert combined_source_data == combined_batch_data


@pytest.mark.parametrize('validation_split', ['train_test'])
def test_datasets(dataset_config, split_datasets):
    match dataset_config['name']:
        case 'UCI_Energy':
            train_size = 692
            test_size = 76
            input_size = 8,
            output_size = 1,
            input_dtype = jnp.floating
            output_dtype = jnp.floating
        case 'FashionMNIST':
            train_size = 60000
            test_size = 10000
            input_size = 784,
            output_size = tuple()
            input_dtype = jnp.floating
            output_dtype = jnp.integer
        case 'CIFAR10':
            train_size = 50000
            test_size = 10000
            input_size = 32, 32, 3
            output_size = tuple()
            input_dtype = jnp.floating
            output_dtype = jnp.integer
        case _:
            raise ValueError(f"Dataset {dataset_config['name']} unconfigured")

    (train_input, train_output), _, (test_input, test_output), input_augmenter = split_datasets
    training_mask = jnp.isfinite(train_input).all(axis=range(1, train_input.ndim))
    assert train_input[training_mask].shape == (train_size, *input_size)
    assert train_output[training_mask].shape == (train_size, *output_size)
    assert test_input.shape == (test_size, *input_size)
    assert test_output.shape == (test_size, *output_size)
    assert jnp.issubdtype(train_input.dtype, input_dtype)
    assert jnp.issubdtype(train_output.dtype, output_dtype)
    assert jnp.issubdtype(test_input.dtype, input_dtype)
    assert jnp.issubdtype(test_output.dtype, output_dtype)

    augmented_input = input_augmenter(train_input)
    assert augmented_input.shape == (train_size, *input_size)
    assert jnp.issubdtype(augmented_input.dtype, input_dtype)


def test_dataset_padding(split_datasets, batch_size, loss_config):
    training_dataset = split_datasets[0]
    unpad_mask = jnp.isfinite(training_dataset[0]).all(axis=range(1, training_dataset[0].ndim))
    training_dataset = tuple(map(lambda x: x[unpad_mask], training_dataset))
    padded_training_dataset = datasets.pad_dataset_for_equal_batches(training_dataset, batch_size)
    shortfall = batch_size - int(((len(training_dataset[0]) / batch_size) % 1) * batch_size)
    for original_array, padded_array in zip(training_dataset, padded_training_dataset):
        assert len(padded_array) == len(original_array) + shortfall
        assert padded_array.shape[1:] == original_array.shape[1:]
        assert (padded_array[:len(original_array)] == original_array).all()
        assert original_array.dtype == padded_array.dtype
        if original_array is training_dataset[1] and loss_config['name'] == 'cross_entropy_loss':
            # NaN appears as different values in different dtypes
            assert (padded_array[-shortfall:] == np.full(1, np.nan, padded_array.dtype)).all()
        else:
            assert jnp.isnan(padded_array[-shortfall:]).all()


def test_dataset_augmentation(split_datasets, rng):
    (training_dataset, _, _), input_augmenter = split_datasets
    training_inputs = training_dataset[0]
    augmented_inputs = input_augmenter(rng(), training_inputs)
    if (training_inputs == augmented_inputs).all():
        # No augmentation performed
        return
    zeros = (augmented_inputs == 0)
    horizontal_zero_bands = zeros.all(axis=1)
    vertical_zero_bands = zeros.all(axis=0)
    assert not horizontal_zero_bands.prod(axis=0).any()
    assert not vertical_zero_bands.prod(axis=1).any()
