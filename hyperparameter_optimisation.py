"""Tools and utilities for hyperparameter optimisation of the experiments."""
import json
import os
import pathlib
import sys
from copy import deepcopy
from datetime import datetime

import numpy as np
import ray
from ray import tune

import config
import train
import util


def parse_config(config):
    master_config = deepcopy(config.pop('master_config'))
    for key in tuple(config.keys()):
        if key.startswith('1-'):
            config[key[2:]] = 1 - config.pop(key)
    overrides = util.flat_to_nested_dict(config)
    return master_config, overrides


def ray_trainable(config, checkpoint_dir=None):
    """Evaluate one `config` under Ray Tune."""
    os.chdir(
        pathlib.Path(__file__).parent.resolve())
    master_config, overrides = parse_config(config)
    train.main(config_dict=master_config,
               config_overrides=overrides)


def get_best_config(directory):
    best_result = tune.Tuner.restore(directory).get_results().get_best_result()
    best_config_file = os.path.join(best_result.log_dir, 'params.json')
    with open(best_config_file, 'r') as best_config_json:
        best_config = json.load(best_config_json)
    return best_config


def save_best_configs(directory):
    for algorithm in os.scandir(directory):
        if not algorithm.is_dir(): continue
        best_config = get_best_config(algorithm.path)
        master_config, overrides = parse_config(best_config)
        util.nested_update(master_config, overrides)

        dataset, optimiser = algorithm.name.split('__')[-3:-1]
        with open(f'configs/optimal/{dataset}__{optimiser}.json', 'w') as saved_config:
            json.dump(master_config, saved_config)


def repeat_best_config(directory, num_repetitions=50):
    best_config = get_best_config(directory)
    best_config['master_config']['log_root'] = directory
    best_config['master_config']['run_group_name'] = 'Best_x50'

    if best_config['master_config']['model']['name'] in ('RosenbrockModel', 'RosenbrockAsLeastSquares'):
        best_config['master_config']['model']['initial_position'] = [1, -1]
        num_repetitions = 1

    ray.init()
    @ray.remote(resources={'accelerator_type:G': 1/(best_config['master_config']['runs_per_gpu'] * ray.cluster_resources()['GPU'])}, max_calls=1)
    def remote_trainer(config):
        result = ray.remote(num_gpus=1/(best_config['master_config']['runs_per_gpu']), max_calls=1)(ray_trainable).remote(config)
        if best_config['master_config']['num_epochs'] == -1:
            _, incomplete = ray.wait([result],
                                    timeout=best_config['master_config']['max_training_time'],
                                    fetch_local=False)
            if incomplete:
                ray.cancel(incomplete[0])
        else:
            ray.get(result)

    try:
        results = [remote_trainer.remote(best_config) for _ in range(num_repetitions)]
        ray.get(results)
    finally:
        ray.shutdown()


def repeat_given_config(num_repetitions=50):
    given_config = config.load_config()

    ray.init()
    @ray.remote(resources={'accelerator_type:G': 1/(given_config['runs_per_gpu'] * ray.cluster_resources()['GPU'])}, max_calls=1)
    def remote_trainer(config):
        result = ray.remote(num_gpus=1/(given_config['runs_per_gpu']), max_calls=1)(ray_trainable).remote(config)
        if given_config['num_epochs'] == -1:
            _, incomplete = ray.wait([result],
                                    timeout=given_config['max_training_time'],
                                    fetch_local=False)
            if incomplete:
                ray.cancel(incomplete[0])
        else:
            ray.get(result)

    try:
        results = [remote_trainer.remote(dict(master_config=given_config)) for _ in range(num_repetitions)]
        ray.get(results)
    finally:
        ray.shutdown()


def run_ablation(directory,
                 config_key,
                 experiment_name,
                 values,
                 repetitions_per_value=50):
    base_config = get_best_config(directory)
    base_config['master_config']['log_root'] = os.path.join(directory, f'{pathlib.Path(directory).name} Sensitivity_{experiment_name}')

    ray.init()
    @ray.remote(num_gpus=1/base_config['master_config']['runs_per_gpu'], max_calls=1)
    def remote_trainer(config): return ray_trainable(config)

    results = []
    try:
        for value in values:
            base_config['master_config']['run_group_name'] = f'{experiment_name}_{value}'
            for _ in range(repetitions_per_value):
                base_config[config_key] = value
                if config_key == 'optimiser.damping_increase_factor':
                    base_config['optimiser.damping_decrease_factor'] = 1 / value
                results.append(remote_trainer.remote(base_config))
        ray.get(results)
    finally:
        ray.shutdown()


def run_amplification_ablation(directory):
    run_ablation(directory,
                 config_key='optimiser.update_amplification',
                 experiment_name='Amplification',
                 values=np.logspace(-1, 1, num=11, base=2))


def run_batch_size_ablation(directory):
    run_ablation(directory,
                 config_key='batch_size',
                 experiment_name='BatchSize',
                 values=(50, 100, 200, 400, 800, 1600, 3200))


def run_initial_damping_ablation(directory):
    run_ablation(directory,
                 config_key='optimiser.initial_damping',
                 experiment_name='InitialDamping',
                 values=np.logspace(-8, 0, num=17, base=10))


def run_lr_clipping_ablation(directory):
    run_ablation(directory,
                 config_key='optimiser.lr_clipping',
                 experiment_name='LRClipping',
                 values=np.logspace(-4, 1, num=11, base=10))


def run_stepping_factor_ablation(directory):
    run_ablation(directory,
                 config_key='optimiser.damping_increase_factor',
                 experiment_name='SteppingFactor',
                 values=np.logspace(0, 2, num=11, base=2))


def run_lr_ablation(directory):
    run_ablation(directory,
                 config_key='optimiser.learning_rate',
                 experiment_name='LearningRate',
                 values=np.logspace(-6, 0, num=19, base=10))


def run_all_ablations(directory):
    run_amplification_ablation(directory)
    run_batch_size_ablation(directory)
    run_initial_damping_ablation(directory)
    run_lr_clipping_ablation(directory)
    run_stepping_factor_ablation(directory)

    run_lr_ablation(directory)


def parse_all_subdirectories(root_directory):
    ray.init()
    @ray.remote
    def remote_parser(directory):
        util.pandas_from_tensorboard(directory)

    try:
        ray.get([
            remote_parser.remote(subdirectory.path)
            for subdirectory in os.scandir(root_directory)
            if subdirectory.is_dir() and subdirectory.name != "Best_x50"])
    finally:
        ray.shutdown()


def construct_search_space(master_config):
    search_space = {'master_config': master_config}
    if master_config['model']['name'] not in ('KroneckerFactoredQuadraticModel',
                                              'RosenbrockModel',
                                              'RosenbrockAsLeastSquares'):
        search_space.update({
            'batch_size': tune.choice((50, 100, 200, 400, 800, 1600, 3200))})

    match master_config['optimiser']['name']:
        case 'sgd':
            search_space.update({
                'optimiser.learning_rate': tune.loguniform(1e-6, 1e-1)})
            if master_config['optimiser'].get('momentum', None):
                search_space.update({
                    '1-optimiser.momentum': tune.loguniform(1e-4, 0.3)})
            if master_config['optimiser'].get('add_decayed_weights', None):
                search_space.update({
                    'optimiser.add_decayed_weights': tune.loguniform(1e-10, 1e-0)})
        case 'adam':
            search_space.update({
                'optimiser.learning_rate': tune.loguniform(1e-6, 1e-0)})
            if master_config['optimiser'].get('eps', None) is not None:
                search_space.update({
                    'optimiser.eps': tune.loguniform(1e-8, 1)})
        case 'SGDQLROptimiser' | 'AdamQLROptimiser':
            if master_config['optimiser'].get('damping_increase_factor', None) is not None:
                search_space.update({
                    'optimiser.initial_damping': tune.loguniform(1e-8, 1),
                    'optimiser.damping_decrease_factor': tune.loguniform(0.5, 1),
                    'optimiser.damping_increase_factor': tune.loguniform(1, 4)})
            if master_config['optimiser'].get('direction_clipping', None) is not None:
                search_space.update({
                    'optimiser.direction_clipping': tune.loguniform(1e-4, 1)})
            if master_config['optimiser'].get('update_norm_clipping', None) is not None:
                search_space.update({
                    'optimiser.update_norm_clipping': tune.loguniform(0.05, 20)})
            if master_config['optimiser'].get('gradient_norm_clipping', None) is not None:
                search_space.update({
                    'optimiser.gradient_norm_clipping': tune.loguniform(0.05, 20)})
            if master_config['optimiser'].get('lr_clipping', None) is not None:
                search_space.update({
                    'optimiser.lr_clipping': tune.loguniform(0.0001, 10)})
            if master_config['optimiser'].get('scaling_envelope_peak', None) is not None:
                search_space.update({
                    'optimiser.scaling_envelope_peak': tune.loguniform(10, 10000)})
        case 'kfac_jax':
            search_space.update({
                'optimiser.initial_damping': tune.loguniform(1e-8, 1)})
            if not master_config['optimiser'].get('use_adaptive_learning_rate'):
                search_space.update({
                    'optimiser.learning_rate': tune.loguniform(1e-6, 1e-0)})
            if not master_config['optimiser'].get('use_adaptive_momentum'):
                search_space.update({
                    '1-optimiser.momentum': tune.loguniform(1e-4, 0.3)})
        case ('KFACwithDynamicKroneckerCorrections'):
            if master_config.get('best_kfac_directory', None):
                kfac_config = tune.Tuner.restore(
                    master_config.pop('best_kfac_directory')
                ).get_results().get_best_result().config
                master_config['optimiser']['initial_damping'] = kfac_config['optimiser.initial_damping']
                master_config['batch_size'] = kfac_config['batch_size']
                search_space.pop('batch_size')
            elif master_config['optimiser']['correction_type'] in ('explicit_override', 'explicit_override_cholesky'):
                search_space.update({
                    'optimiser.initial_learned_correction': tune.loguniform(1e0, 1e6)})
            else:
                search_space.update({
                    'optimiser.initial_damping': tune.loguniform(1e-8, 1)})
            search_space.update({
                'optimiser.correction_optimiser.learning_rate': tune.loguniform(1e-6, 1e-0)})
            if 'weight_decay' in master_config['optimiser']['correction_optimiser']:
                search_space.update({
                    'optimiser.correction_optimiser.weight_decay': tune.loguniform(1e-9, 1e6)})
        case 'BaydinSGD':
            search_space.update({
                'optimiser.initial_learning_rate': tune.loguniform(1e-6, 1e-0),
                'optimiser.hypergradient_learning_rate': tune.loguniform(1e-6, 1e-0)})

    if master_config['dataset']['name'] in ('PennTreebank', 'PennTreebankForGPT2'):
        search_space.update({
            'batch_size': tune.choice((5, 10, 20, 35, 50, 100, 200)),
            'dataset.subsequence_length': tune.choice((10, 20, 30, 40, 50, 60, 70, 80, 90, 100))})
        if 'optimiser.learning_rate' in search_space:
            search_space['optimiser.learning_rate'] = tune.loguniform(1e-4, 1e2)

    return search_space


def construct_experiment_name():
    experiment_name = datetime.now().isoformat()
    for shell_arg in sys.argv:
        if shell_arg.endswith('.yaml'):
            experiment_name += '__' + pathlib.Path(shell_arg).stem
    return experiment_name


def run_hpo_algorithm():
    master_config = config.load_config()
    max_training_time = master_config.get('max_training_time')
    runs_per_gpu = master_config.get('runs_per_gpu')
    num_samples = master_config.get('num_samples')
    tuning_metric = master_config.get('tuning_metric')
    scratch_dir = master_config.get('scratch_dir')
    tuner_name = construct_experiment_name()

    ray.init()
    tuner = tune.Tuner(
        trainable=tune.with_resources(
            ray_trainable,
            resources={'gpu': 1/runs_per_gpu}),
        tune_config=tune.TuneConfig(
            metric=tuning_metric,
            mode='min',
            num_samples=num_samples,
            reuse_actors=False,  # Shouldn't be needed, but memory leaks otherwise
            scheduler = tune.schedulers.ASHAScheduler(
                time_attr='time_total_s',
                max_t=max_training_time)),
        run_config=ray.air.config.RunConfig(
            name=tuner_name,
            local_dir=scratch_dir,
            stop=None,
            log_to_file=True),
        param_space=construct_search_space(master_config))
    tuner.fit()
    ray.shutdown()
    repeat_kwargs = {}
    if master_config['dataset']['name'] in ('PennTreebank', 'PennTreebankForGPT2'):
        repeat_kwargs['num_repetitions'] = 10
    repeat_best_config(os.path.join(scratch_dir, tuner_name), **repeat_kwargs)
    # util.pandas_from_tensorboard(os.path.join(scratch_dir, tuner_name))


def run_all_best_config_evaluations(directories):
    if not isinstance(directories, (tuple, list)):
        directories = [subdirectory.path
                       for subdirectory in os.scandir(directories)
                       if subdirectory.is_dir()]
    for directory in directories:
        print(directory)
        repeat_best_config(directory)


if __name__ == '__main__':
    if '--SKIP_HPO' in sys.argv:
        sys.argv.remove('--SKIP_HPO')
        repeat_given_config()
    else:
        run_hpo_algorithm()
