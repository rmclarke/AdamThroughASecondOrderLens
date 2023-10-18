import json
import os

import pytest
from ray import tune

import hyperparameter_optimisation as hpo
import train


@pytest.mark.parametrize('directory',
                         (directory for directory in os.scandir('/scratch/dir/ImprovingKFAC/ray/')
                          if os.path.exists(os.path.join(directory, 'tuner.pkl'))))
def test_best_config_identification(directory):
    results_grid = tune.Tuner.restore(directory.path).get_results()
    ray_best_result = results_grid.get_best_result()

    verified_best_result = None
    verified_best_loss = float('inf')
    for result in results_grid:
        if result.metrics_dataframe is None:
            continue
        if result.metrics_dataframe['validation_loss'].iloc[-1] < verified_best_loss:
            verified_best_result = result
            verified_best_loss = verified_best_result.metrics_dataframe['validation_loss'].iloc[-1]

    assert verified_best_result == ray_best_result

    ray_best_config_path = os.path.join(ray_best_result.log_dir, 'params.json')
    with open(ray_best_config_path, 'r') as ray_best_config_json:
        ray_best_config = json.load(ray_best_config_json)
    assert ray_best_config == hpo.get_best_config(directory)


def test_construct_search_space(optimiser_config, monkeypatch):
    search_space = hpo.construct_search_space(optimiser_config)
    assert search_space.pop('master_config', False)
    assert search_space.pop('batch_size', False)

    matched_case = False
    if optimiser_config['optimiser']['name'] in ('sgd', 'adam'):
        matched_case = True
        assert search_space.pop('optimiser.learning_rate', False)
        assert bool(search_space.pop('optimiser.add_decayed_weight', False)) == bool('add_decayed_weight' in optimiser_config['optimiser'])
        if 'momentum' in optimiser_config['optimiser']:
            one_minus_momentum = search_space.pop('1-optimiser.momentum')
            overrides = {}
            monkeypatch.setitem(train, 'main', lambda _, config_overrides: overrides.update(config_overrides))
            full_search_space = hpo.construct_search_space(optimiser_config)
            param_config = {k: (v.sample() if k != 'master_config' else v)
                            for k, v  in full_search_space.items()}
            hpo.ray_trainable(param_config)
            assert overrides['optimiser.momentum'] == 1 - one_minus_momentum
            assert '1-optimiser.momentum' not in overrides
    if optimiser_config['optimiser']['name'] in ('SGDQLROptimiser', 'AdamQLROptimiser'):
        assert bool(search_space.pop('optimiser.initial_damping', False)) == bool(optimiser_config['optimiser']['damping_adjustment_factor'])
        matched_case = True
    if optimiser_config['optimiser']['name'] == 'kfac_jax':
        assert search_space.pop('optimiser.initial_damping', False)
        matched_case = True
    if optimiser_config['optimiser']['name'].endswith('withDynamicKroneckerCorrections'):
        assert bool(search_space.pop('optimiser.initial_learned_correction', False)) == (optimiser_config['optimiser']['correction_type'] in ('explicit_override', 'explicit_override_cholesky'))
        assert bool(search_space.pop('optimiser.initial_damping', False)) == (optimiser_config['optimiser']['correction_type'] in ('implicit', 'explicit', 'explicit_cholesky'))
        assert search_space.pop('optimiser.correction_optimiser.learning_rate', False)
        assert bool(search_space.pop('optimiser.correction_optimiser.weight_decay', False)) == bool(optimiser_config['optimiser']['correction_optimiser'].get('weight_decay', False))

        monkeypatch.setitem(optimiser_config, 'best_kfac_directory', '/scratch/dir/ImprovingKFAC/ray/2023-03-10T18:05:10.923438__fashion_mnist__KFAC__ASHA')
        revised_search_space = hpo.construct_search_space(optimiser_config)
        assert 'batch_size' not in revised_search_space
        assert optimiser_config.pop('batch_size', False)
        assert optimiser_config['optimiser'].pop('initial_damping', False)
        matched_case = True

    assert matched_case
    assert not search_space
