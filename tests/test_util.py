from datetime import datetime

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import ray

import util


# def test_pandas_from_tensorboard():
#     # Tested as part of logging in training
#     pass


def test_nested_update():
    initial_dict = dict(a='a',
                        b='b',
                        c=dict(a='A',
                               b='B',
                               c=dict(a='aa',
                                      b='bb')),
                        d='d')
    update_dict = dict(b='b_Updated',
                       c=dict(a='A_Updated',
                              c=dict(b='bb_Updated')))
    target_dict = dict(a='a',
                       b='b_Updated',
                       c=dict(a='A_Updated',
                              b='B',
                              c=dict(a='aa',
                                     b='bb_Updated')),
                       d='d')
    util.nested_update(initial_dict, update_dict)
    assert target_dict == initial_dict


def test_bootstrap_sample(data_length=300,
                          num_datasets=10,
                          num_samples=250):
    indices = util.bootstrap_sample(data_length, num_datasets, num_samples)
    chex.assert_shape(indices, (num_datasets, num_samples))
    assert (indices >= 0).all()
    assert (indices < data_length).all()
    assert not (indices == indices.flat[0]).all()


def test_maybe_initialise_determinism():
    baseline_state = np.random.get_state()

    util.maybe_initialise_determinism()
    chex.assert_trees_all_equal(np.random.get_state(), baseline_state)

    util.maybe_initialise_determinism(
        seed=int(datetime.now().timestamp()))
    assert not np.allclose(baseline_state[1], np.random.get_state()[1])


def test_cosine_similarity():
    x = np.random.randn(100)
    y = np.random.randn(100)

    assert -1 <= util.cosine_similarity(x, y) <= 1

    for singleton in (x, y):
        assert np.allclose(util.cosine_similarity(singleton, singleton), 1)
        assert np.allclose(util.cosine_similarity(singleton, -singleton), -1)


def test_flat_to_nested_dict():
    nested_dict = dict(a='a',
                       b='b',
                       c=dict(ca='ca',
                              cb='cb',
                              cc=dict(a='aa',
                                      b='bb')))
    flat_dict = {'a': 'a',
                 'b': 'b',
                 'c.ca': 'ca',
                 'c.cb': 'cb',
                 'c.cc.a': 'aa',
                 'c.cc.b': 'bb'}
    assert util.flat_to_nested_dict(flat_dict) == nested_dict


def test_ray_sessions():
    assert not util.in_monitored_ray_session()
    assert not util.in_any_ray_session()
    try:
        ray.init()
        assert not util.in_monitored_ray_session()
        assert util.in_any_ray_session()
    finally:
        ray.shutdown()
    assert not util.in_monitored_ray_session()
    assert not util.in_any_ray_session()


def test_round_sig_figs():
    sample_data = pd.Series([
        1234.5678, 1234.567, 1234.56, 1234.5, 1234.0, 1234, 1230, 1200, 1000,
        234.5678, 234.567, 234.56, 234.5, 234.0, 234, 230, 200,
        34.5678, 34.567, 34.56, 34.5, 34.0, 34, 30,
        4.5678, 4.567, 4.56, 4.5, 4.0, 4,
        0.5678, 0.567, 0.56, 0.5, 0.0, 0,
        0.678, 0.67, 0.6,
        0.78, 0.7,
        0.8])
    assert (util.round_sig_figs(sample_data, 3) == pd.Series([
        1230.0, 1230.0, 1230.0, 1230.0, 1230.0, 1230, 1230, 1200, 1000,
        235.0, 235.0, 235.0, 234.0, 234.0, 234, 230, 200,
        34.6, 34.6, 34.6, 34.5, 34.0, 34, 30,
        4.57, 4.57, 4.56, 4.5, 4.0, 4,
        0.568, 0.567, 0.56, 0.5, 0.0, 0,
        0.678, 0.67, 0.6,
        0.78, 0.7,
        0.8])).all()
    assert (util.round_sig_figs(sample_data, 2) == pd.Series([
        1200.0, 1200.0, 1200.0, 1200.0, 1200.0, 1200, 1200, 1200, 1000,
        230.0, 230.0, 230.0, 230.0, 230.0, 230, 230, 200,
        35.0, 35.0, 35.0, 34.0, 34.0, 34, 30,
        4.6, 4.6, 4.6, 4.5, 4.0, 4,
        0.57, 0.57, 0.56, 0.5, 0.0, 0,
        0.68, 0.67, 0.6,
        0.78, 0.7,
        0.8])).all()
    assert (util.round_sig_figs(sample_data, 1) == pd.Series([
        1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000, 1000, 1000, 1000,
        200.0, 200.0, 200.0, 200.0, 200.0, 200, 200, 200,
        30.0, 30.0, 30.0, 30.0, 30.0, 30, 30,
        5.0, 5.0, 5.0, 4.0, 4.0, 4,
        0.6, 0.6, 0.6, 0.5, 0.0, 0,
        0.7, 0.7, 0.6,
        0.8, 0.7,
        0.8])).all()


def test_all_combinations():
    source = tuple(range(10))
    combinations = set(util.all_combinations(source))
    for inclusion_bitmask in range(2**10):
        trial_combination = tuple(number for number, bit in zip(source, range(10))
                                  if inclusion_bitmask & 2**bit)
        assert trial_combination in combinations
        combinations.remove(trial_combination)
    assert not combinations


def test_top_1_accuracy(rng):
    logits = jax.random.normal(rng(), (100, 10))
    targets = logits.argmax(axis=1)
    assert util.top_1_accuracy(logits, targets) == 1
    finite_mask = jax.random.bernoulli(rng(), shape=(100,))
    assert util.top_1_accuracy(logits, targets, finite_mask) == 1

    targets[50:] = -1
    assert util.top_1_accuracy(logits, targets) == 0.5
    finite_mask = jnp.ones_like(finite_mask)
    assert util.top_1_accuracy(logits, targets, finite_mask) == 0.5
    finite_mask[50:] = 0
    assert util.top_1_accuracy(logits, targets, finite_mask) == 1
    finite_mask[:50] = 0
    finite_mask[50:] = 1
    assert util.top_1_accuracy(logits, targets, finite_mask) == 0
    finite_mask[:] = 1
    finite_mask[25:75] = 0
    assert util.top_1_accuracy(logits, targets, finite_mask) == 0.5
