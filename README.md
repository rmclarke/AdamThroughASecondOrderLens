This repository contains code for the paper _Adam through a Second-Order Lens_ ,
submitted to ICLR 2024.


# Installation

Our complete development environment under Python 3.10 is specified in
`local_requirements.txt`, with a list of top-level requirements given in
`Pipfile`. In theory, `pipenv install` in a fresh virtual environment will set
everything up; in practice, JAX in particular may need manual intervention
depending on your local CUDA and cuDNN versions.

At the time of writing, we depend on a bugfix to the KFAC-JAX library, which is
specified in `kfac_jax.patch`. This can be applied from the project root with
``` shell
$ patch -p0 -i kfac_jax.patch
```

Datasets are not bundled with the repository, so before first use they will need
to be downloaded by calling the constructors with `download=True`.


# Running
Each dataset and algorithm is specified by a YAML configuration file in
`configs/`, where `AdamQLR_Damped.yaml` is the _AdamQLR (Tuned)_
algorithm described in our paper, and `AdamQLR_NoHPO.yaml` is the _AdamQLR
(Untuned)_ setting. To perform a single training run, simply pass
the corresponding files to `train.py` with the `-c` flag, e.g.:
``` shell
$ python train.py -c ./configs/fashion_mnist.yaml ./configs/AdamQLR_Damped.yaml
```

A complete hyperparameter optimisation routine, including 50 repetitions of the
best hyperparameters found, can be performed by calling
`hyperparameter_optimisation.py` with the corresponding configuration files:
``` shell
$ python hyperparameter_optimisation.py -c ./configs/fashion_mnist.yaml ./configs/AdamQLR_Damped.yaml ./configs/ASHA.yaml
```
This same file also contains helper functions for running sensitivity studies.
Hyperparameter optimisation runs based on overall runtime rather than number of
epochs may be performed by substituting `./configs/ASHA_time_training.yaml` or
`./configs/ASHA_time_validation.yaml` in place of `./configs/ASHA.yaml`.

To replicate all our experimental results, the various `run_*.sh*` scripts may
be useful.

# Analysis
Logs are produced by Tensorboard in a `runs/` directory by default; the paths
can be changed with the config/command-line flag `--log-root`.

All our experimental plots are produced using `paper_plots.py`, though you may
need to update the paths to match your local configuration.
