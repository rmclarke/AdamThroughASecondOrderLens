"""Global fixtures for pytest."""

import os
import pathlib
from copy import deepcopy
from datetime import datetime
from itertools import product

import chex
import jax
import jax.numpy as jnp
import haiku as hk
import pytest
import ruamel.yaml
from tensorboardX import SummaryWriter

import datasets
import models
import train
import util

# Setup a safe, YAML 1.2 parser
_yaml = ruamel.yaml.YAML(typ='safe', pure=True)


def scan_configs(search_key, include_extras=False):
    """Identify all configs containing `search_key`."""
    extras = [pathlib.Path('./configs/addCorrectedKFACWeightDecay.yaml')]
    for subpath in os.scandir('./configs/'):
        subpath = pathlib.Path(subpath)
        if subpath in extras: continue
        if not subpath.is_file(): continue

        with open(subpath, 'r') as config_file:
            config_dict = _yaml.load(config_file)
        if search_key not in config_dict: continue

        if include_extras:
            if subpath.name.startswith('KFACwithDynamicKroneckerCorrection'):
                for extras_combination in util.all_combinations(extras):
                    yield subpath, *extras_combination
            else:
                yield subpath,
        else:
            yield subpath


@pytest.fixture(scope='session', autouse=True)
def clear_gpu():
    """Attempt to recover from any CUDA errors by clearing JAX memory use."""
    jax.clear_backends()
    for device in jax.devices():
        for buffer in device.client.live_buffers():
            buffer.delete()


@pytest.fixture
def block_rng_splits(monkeypatch):
    """Convert all jax.random.split() calls into non-ops on the RNG."""
    monkeypatch.setattr('jax.random.split', lambda rng, num=2: [rng for _ in range(num)])
    monkeypatch.setattr('jax.random.permutation', lambda _, length: jnp.arange(length))


@pytest.fixture
def logger(tmp_path):
    """Create dummy TensorBoard logging object."""
    with SummaryWriter(log_dir=tmp_path) as logger:
        yield logger


@pytest.fixture
def rng():
    """Reusable random number generator."""
    base_rng = jax.random.PRNGKey(
        int(datetime.now().timestamp() * 1e6))

    def generate_new_rng():
        nonlocal base_rng
        base_rng, new_rng = jax.random.split(base_rng)
        return new_rng
    return generate_new_rng


@pytest.fixture(params=['train_test', 'train_val_test', 'config_splits'])
def validation_split(request):
    """Choose between configured validation split and forced extremes."""
    return request.param


@pytest.fixture(params=product(scan_configs('model', include_extras=True),
                               scan_configs('optimiser', include_extras=True)))
def valid_config_pair(request):
    """Generate valid pairs of dataset and optimiser configuration files."""
    return [file for file in sum(map(list, request.param), [])]


@pytest.fixture(params=scan_configs('model'),
                ids=[file.name[:-5] for file in scan_configs('model')])
def compatible_model_loss_dataset_kwargs(request, monkeypatch):
    """Generate sets of compatible subconfigs."""
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    with open(request.param, 'r') as config_file:
        config_dict = _yaml.load(config_file)
    return (config_dict['model'],
            config_dict['loss'],
            config_dict['dataset'],
            config_dict.get('forward_pass_extra_kwargs', {}))


@pytest.fixture
def model_config(compatible_model_loss_dataset_kwargs):
    """Model configuration dictionary."""
    return compatible_model_loss_dataset_kwargs[0]


@pytest.fixture
def model(model_config):
    """Each model specified in a config file."""
    return models.create_model(**model_config)


@pytest.fixture
def model_params_and_state(model,
                           sample_batch,
                           rng,
                           training_forward_pass_kwargs,
                           model_config):
    """Initialise parameters and state for models."""
    chex.clear_trace_counter()
    # Match operation we need to test against in TrainingState.init
    # (see test_train.test_training_state_init_and_advance())
    sub_rng1, sub_rng2 = jax.random.split(rng())
    if model_config['name'].startswith('ResNet'):
        # ResNets use BatchNorm layers, so can't be initialised with
        # is_training=False
        training_forward_pass_kwargs['is_training'] = True
    model_params, model_state = model.init(sub_rng1,
                                           sample_batch[0],
                                           **training_forward_pass_kwargs)
    model_params = jax.tree_util.tree_map(
        # Need to avoid calling rng() again, otherwise the real and test code
        # get out of step
        lambda x: x + 3e-3*jax.random.normal(sub_rng2, x.shape),
        model_params)
    return model_params, model_state


@pytest.fixture
def loss_config(compatible_model_loss_dataset_kwargs):
    return compatible_model_loss_dataset_kwargs[1]


@pytest.fixture
def loss_fn(loss_config):
    """Fixture generating each loss function specified in a config file."""
    return models.create_loss(**loss_config)


@pytest.fixture(params=[True, False],
                ids=['Padded', 'Unpadded'])
def dataset_config(compatible_model_loss_dataset_kwargs,
                   validation_split,
                   batch_size,
                   request,
                   monkeypatch):
    """Dataset configuration dictionary."""
    dataset_dict = compatible_model_loss_dataset_kwargs[2]
    if validation_split == 'train_val_test':
        monkeypatch.setitem(dataset_dict, 'validation_proportion', 0.2)
    elif validation_split == 'train_test':
        monkeypatch.setitem(dataset_dict, 'validation_proportion', 0)

    if request.param:
        monkeypatch.setitem(dataset_dict, 'pad_to_equal_training_batches', batch_size)

    return compatible_model_loss_dataset_kwargs[2]


@pytest.fixture
def split_datasets(dataset_config):
    """Fixture generating each dataset specified in a config file."""
    return datasets.make_split_datasets(**dataset_config)


@pytest.fixture
def batch_size():
    """Return a modifiable batch size."""
    return 128


@pytest.fixture
def sample_batch(split_datasets, batch_size, rng):
    """First batch of generated data from a dataset."""
    return next(datasets.make_batches(
        split_datasets[0],
        batch_size,
        rng()))


@pytest.fixture
def forward_pass_extra_kwargs(compatible_model_loss_dataset_kwargs):
    """Extra flag names required by a model at forward-pass time."""
    return compatible_model_loss_dataset_kwargs[3]


@pytest.fixture(params=[True, False],
                ids=['Training', 'Validation/Test'])
def training_forward_pass_kwargs(request, forward_pass_extra_kwargs):
    """Extra flag dict required by a model at forward-pass time."""
    return train.construct_forward_pass_extra_kwargs(
        forward_pass_extra_kwargs,
        is_training=request.param)


@pytest.fixture
def forward_pass_fn(model, loss_fn, training_forward_pass_kwargs):
    """Generate a function for computing forward passes of models."""
    return lambda params, model_state, batch: train.forward_pass(
        model=model,
        params=params,
        model_state=model_state,
        loss_function=loss_fn,
        batch=batch,
        **training_forward_pass_kwargs)


@pytest.fixture(params=scan_configs('optimiser', include_extras=True),
                ids=['-'.join(f.name[:-5] for f in files) for files in scan_configs('optimiser', include_extras=True)])
def optimiser_config(request):
    """Fixture returning each specified optimiser config."""
    loaded_config = {}
    for config_path in request.param:
        with open(config_path, 'r') as config_file:
            util.nested_update(loaded_config, _yaml.load(config_file))
    return loaded_config


@pytest.fixture
def optimiser(optimiser_config, forward_pass_fn):
    """Fixture generating each optimiser specified in a config file."""
    return train.create_optimiser(
        forward_pass_fn=forward_pass_fn,
        **optimiser_config['optimiser'])


@pytest.fixture
def optimiser_state(optimiser, rng, model_params_and_state, sample_batch):
    """Initialise optimiser state."""
    model_params, model_state = model_params_and_state
    # Match operation we need to test against in TrainingState.init
    # (see test_train.test_training_state_init_and_advance())
    rng = jax.random.split(rng())[1]
    return optimiser.init(model_params,
                          rng,
                          sample_batch,
                          func_state=model_state)


@pytest.fixture
def one_optimiser_step(model_params_and_state,
                       sample_batch,
                       optimiser,
                       optimiser_state,
                       rng):
    """Perform one optimiser step."""
    # KFAC optimiser causes state dicts to be deleted, so must make sure
    # we aren't using the same copy as was created for another test/fixture
    model_params, model_state = deepcopy(model_params_and_state)
    optimiser_state = deepcopy(optimiser_state)
    return optimiser.step(
        model_params,
        optimiser_state,
        func_state=model_state,
        batch=sample_batch,
        rng=rng(),
        global_step_int=0)


@pytest.fixture
def training_state(model, optimiser):
    """Construct a raw TrainingState object."""
    return train.TrainingState(model, optimiser)
