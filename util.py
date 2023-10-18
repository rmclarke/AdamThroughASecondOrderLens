"""Helper utilities and functions."""

import os
from itertools import combinations

import chex
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import jax.numpy as jnp
import optax
import ray
from ray import tune
from tbparse import SummaryReader

import util


def nested_update(source_dict, update_dict):
    """Recursively update each level of `source_dict` with the contents of
    `update_dict`.
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in source_dict:
            nested_update(source_dict[key], value)
        else:
            source_dict[key] = value


def bootstrap_sample(data_length, num_datasets, num_samples=None):
    """Bootstrap sample, generating `num_datasets` sample sets of `num_samples`
    each, returning the indices of the sample.
    """
    if num_samples is None:
        num_samples = data_length
    return np.random.choice(data_length,
                            replace=True,
                            size=(num_datasets, num_samples))


def maybe_initialise_determinism(seed=None):
    """Initialiser function for main and worker threads, using the provided
    random `seed` if it is set.
    """
    if seed is None:
        return

    # if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
    #     # or :16:8 if memory becomes an issue
    #     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    np.random.seed(seed)


def cosine_similarity(x, y):
    return x.dot(y) / (jnp.linalg.norm(x) * jnp.linalg.norm(y))


def pandas_from_tensorboard(root_directory):
    store_kwargs = dict(path=os.path.join(root_directory, 'pandas_store.h5'))
    try:
        # Try opening in read mode first to permit concurrent access
        with pd.HDFStore(mode='r', **store_kwargs) as store:
            if 'scalars' in store:
                return store['scalars']
    except FileNotFoundError:
        # Data doesn't exist, so open in write mode
        with pd.HDFStore(**store_kwargs) as store:
                scalar_data = SummaryReader(root_directory,
                                            pivot=False,
                                            extra_columns={'dir_name', 'wall_time'}).scalars
                store['scalars'] = scalar_data
                return scalar_data


def plot_from_ray_directory(directory, *args):
    ax = plt.gca()
    for directory in os.scandir(directory):
        tuner = tune.Tuner.restore(directory.path)
        result_grid = tuner.get_results()
        result_grid.get_best_result().metrics_dataframe.plot(*args, ax=ax, label=directory.name)
    plt.yscale('log')
    plt.show()


def in_monitored_ray_session():
    from ray.train._internal.session import _session_v2 as train_session
    from ray.tune.trainable.session import _session_v2 as tune_session
    return train_session or tune_session

def in_any_ray_session():
    return in_monitored_ray_session() or ray.is_initialized()


def flat_to_nested_dict(data):
    overrides = {}
    for key, value in data.items():
        split_keys = key.split('.')
        parent_dict = overrides
        for split_key in split_keys[:-1]:
            parent_dict = parent_dict.setdefault(split_key, {})
        parent_dict[split_keys[-1]] = value
    return overrides


def nested_to_flat_dict(data, prefix=''):
    flat_dict = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flat_dict.update(nested_to_flat_dict(value, prefix + key + '.'))
        else:
            flat_dict[prefix + key] = value
    return flat_dict


def except_ray_jit_only_once(func):
    """Assert that the decorated function is JITted only once, unless we're in a Ray environment."""
    if util.in_any_ray_session():
        return func
    else:
        return chex.assert_max_traces(func, n=1)


def round_sig_figs(data, num_sig_figs):
    """Round `data` to `num_sig_figs`."""
    return data.apply(
        lambda x: (
            round(x,
                  (num_sig_figs - np.floor(np.log10(np.abs(x))) - 1).astype(int))
            if x != 0 else x))


def maybe_print(*args, **kwargs):
    if not in_any_ray_session():
        print(*args, **kwargs)


def extract_best_data_from_symlinks(root_directory):
    for algorithm_directory in os.scandir(root_directory):
        if not (algorithm_directory.is_symlink() and algorithm_directory.is_dir()):
            continue
        target_dir = os.path.join(os.path.realpath(algorithm_directory), 'Best_x50')
        if not os.path.exists(target_dir):
            continue
        algorithm_name = algorithm_directory.name.split('__')[-2]
        os.symlink(target_dir,
                   os.path.join(root_directory, algorithm_name))
        os.unlink(algorithm_directory)


def all_combinations(iterable):
    """Generate a list of all (sub-)combinations of `iterable`."""
    for length in range(len(iterable) + 1):
        for combination in combinations(iterable, length):
            yield combination


def top_1_accuracy(logits, targets):
    # Padded data points will give NaN logits, which will automatically give
    # False values in the accuracy array
    return jnp.mean(logits.argmax(axis=1) == targets)


def rosenbrock(x, y, a, b):
    return (a - x)**2 + b*(y - x**2)**2


def extract_individual_rosenbrock_symlinks(root_directory):
    """Transform `directory` symlinks from pointers to 50 runs
    to pointers to the first of those runs."""
    for algorithm_symlink in os.scandir(root_directory):
        true_algorithm_path = os.path.realpath(algorithm_symlink)
        first_run = next(os.scandir(true_algorithm_path))
        os.unlink(algorithm_symlink)
        os.mkdir(algorithm_symlink)
        os.symlink(first_run, os.path.join(algorithm_symlink, first_run.name))


def linear_warmup_cosine_decay_schedule(peak_steps, end_steps):
    """Returns a scheduling function rising from 0 to 1 at `break_point`, then
    cosine decaying to 0 at `end_steps`."""
    warmup_schedule = optax.linear_schedule(
        init_value=0,
        end_value=1,
        transition_steps=peak_steps)
    decay_schedule = optax.cosine_decay_schedule(
        init_value=1,
        decay_steps=(end_steps - peak_steps))
    return optax.join_schedules(
        schedules=(warmup_schedule, decay_schedule),
        boundaries=(peak_steps,))
