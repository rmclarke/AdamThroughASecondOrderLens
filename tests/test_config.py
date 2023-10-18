import chex
import ruamel.yaml
_yaml = ruamel.yaml.YAML(typ='safe', pure=True)

import sys

import config
import util


def test_config_read_types(monkeypatch, valid_config_pair):
    monkeypatch.setattr(
        'sys.argv',
        sys.argv[0:1] + ['-c'] + [str(config) for config in valid_config_pair])
    expected_types = dict(
        config=list,
        model=dict,
        dataset=dict,
        optimiser=dict,
        loss=dict,
        device=str,
        batch_size=int,
        num_epochs=int,
        validation_proportion=(float, int),
        seed=int,
        save_state=str,
        load_state=str,
        log_root=str,
        run_group_name=str,
        run_name=str,
        ray_search_space_spec=str,
        time_s=int,
        runs_per_gpu=int,
        tuning_metric=str,
        forward_pass_extra_kwargs=list,
        max_training_time=int)
    expected_subkeys = dict(
        model=('name',),
        dataset=('name',),
        optimiser=('name',),
        loss=('name',))
    for key, value in config.load_config().items():
        if value is not None:
            assert isinstance(value, expected_types[key])
        if isinstance(value, dict):
            for subkey in expected_subkeys[key]:
                assert subkey in value


def test_config_loading(monkeypatch, valid_config_pair):
    monkeypatch.setattr(
        'sys.argv',
        sys.argv[0:1] + ['-c'] + [str(config) for config in valid_config_pair])
    yaml_config = {}
    for config_path in valid_config_pair:
        with open(config_path, 'r') as config_text:
            util.nested_update(yaml_config, _yaml.load(config_text))
    # Account for default values which won't be specified in the config
    yaml_config.setdefault('validation_proportion', 0)
    yaml_config.setdefault('seed', None)
    yaml_config.setdefault('save_state', None)
    yaml_config.setdefault('load_state', None)
    yaml_config.setdefault('log_root', './runs')
    yaml_config.setdefault('run_name', None)
    yaml_config.setdefault('run_group_name', 'Untitled')
    yaml_config.setdefault('ray_search_space_spec', None)
    yaml_config.setdefault('time_s', None)
    yaml_config.setdefault('runs_per_gpu', None)
    yaml_config.setdefault('tuning_metric', None)
    yaml_config.setdefault('forward_pass_extra_kwargs', [])
    yaml_config.setdefault('best_kfac_directory', None)

    loaded_config = config.load_config()
    assert yaml_config == loaded_config


def test_log_directory_creation(monkeypatch):
    config_dict = dict(run_name='Test Run Name',
                       log_root='Test Log Root',
                       run_group_name='Test Run Group Name')
    log_directory = config.log_directory(config_dict)
    assert log_directory.startswith('Test Log Root/Test Run Group Name/')
    assert log_directory.endswith('Test Run Name')
    assert config_dict == {}
    config_dict = dict(run_name=None,
                       log_root='Test Log Root',
                       run_group_name='Test Run Group Name')
    log_directory = config.log_directory(config_dict)
    assert log_directory.startswith('Test Log Root/Test Run Group Name')
    assert not log_directory.endswith('Test Run Name')
    assert config_dict == {}
