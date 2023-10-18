"""Functions and utilities for working with experiment configurations."""

import configargparse
import datetime
import os
import ruamel.yaml

import util

# Setup a safe, YAML 1.2 parser
_yaml = ruamel.yaml.YAML(typ='safe', pure=True)


class ConfigFileAction(configargparse.Action):
    """Argparse action to process a config file from its path, allowing
    multiple files to be specified.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        for config_file in values:
            with open(config_file, 'r') as config_text:
                config_updates = _yaml.load(config_text)
            # Can't treat a namespace as a dictionary, so deal with the first
            # level of nesting ourselves
            for key, new_value in config_updates.items():
                if isinstance(getattr(namespace, key, None), dict):
                    util.nested_update(getattr(namespace, key), new_value)
                else:
                    setattr(namespace, key, new_value)


def load_config(config_overrides={}):
    """Parse command line arguments and config files, making
    interpretations/adjustments as needed.
    """
    parser = configargparse.ArgParser()

    parser.add('-c', '--config',
               action=ConfigFileAction,
               nargs='*',
               help='config file path')
    parser.add('--model',
               type=_yaml.load,
               help="Yaml specifying parameters for the model. See example configs")
    parser.add('--dataset',
               type=_yaml.load,
               help="Yaml specifying parameters for the dataset. See example configs")
    parser.add('--optimiser',
               type=_yaml.load,
               help="Yaml specifying parameters for the optimiser. See example configs")
    parser.add('--loss',
               type=_yaml.load,
               help="Yaml specifying parameters for the loss. See example configs")
    parser.add('--batch_size',
               type=int,
               help='Size of the batch to use in training')
    parser.add('--num_epochs',
               type=int,
               help='How many epochs of data to consider')
    parser.add('--seed',
               type=int,
               default=None,
               help='Force an initial seed for the random number generators')
    parser.add('-S', '--save_state',
               type=str,
               default=None,
               help='File in which to save final training state')
    parser.add('-s', '--load_state',
               type=str,
               default=None,
               help='File from which to load initial training state')
    parser.add('-l', '--log_root',
               type=str,
               help='Root directory for logging run data',
               default='./runs')
    parser.add('-g', '--run_group_name',
               type=str,
               help='run group name',
               default='Untitled')
    parser.add('-n', '--run_name',
               type=str,
               help='run name')
    parser.add('-R', '--ray_search_space_spec',
               type=str,
               help='Search space specifier for Ray optimisation',
               default=None)
    parser.add('--time_s',
               type=int,
               default=None,
               help='When using Ray tune, how many seconds to allow ASHA for tuning')
    parser.add('--runs_per_gpu',
               type=int,
               default=None,
               help='How many runs Ray should attempt to run on a single gpu')
    parser.add('--tuning_metric',
               type=str,
               help='Metric with respect to which Ray will tune',
               default=None)
    parser.add('--forward_pass_extra_kwargs',
               type=_yaml.load,
               default=[],
               help='List of additional flags which the model forward-pass function requires')
    parser.add('--best_kfac_directory',
               type=str,
               help='Path to base KFAC experiments when calling hyperparameter_optimiser.py',
               default=None)
    parser.add('--recurrent_model_state',
               type=_yaml.load,
               default=False,
               help='Request a new model state for each epoch/dataset')
    args = parser.parse_args()

    config_dict = dict(args._get_kwargs())
    config_dict.pop('config')

    util.nested_update(config_dict, config_overrides)
    return config_dict


def log_directory(config_dict):
    """Compute, create and return the appropriate log directory, removing
    consumed components of the configuration.
    """
    run_prefix = datetime.datetime.now().isoformat()
    run_suffix = config_dict.pop('run_name')
    log_root = config_dict.pop('log_root')
    if run_suffix is None:
        run_name = run_prefix
    else:
        run_name = ' '.join((run_prefix, run_suffix))
    return os.path.join(log_root, config_dict.pop('run_group_name'), run_name)
