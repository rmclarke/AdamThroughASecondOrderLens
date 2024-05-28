#!/bin/bash

optimisers=(
    'SGD'
    'AdamQLR_Damped_NoLRClipping'
    'Adam'
    'KFAC'
    'SGDmwd'
    'AdamQLR_Damped'
    'KFAC_Unadaptive'
    'Adam_TunedEpsilon'
    'BaydinSGD'
)

datasets=(
    'uci_energy'
    'uci_protein'
    'fashion_mnist'
    'svhn'
    'cifar-10_15mins'
    'penn_treebank_gpt2_reset'
)

for dataset in ${datasets[@]}
do
    for optimiser in ${optimisers[@]}
    do
        echo "================================================================================"
        echo "================================================================================"
        echo "================================================================================"
        echo "Starting ${dataset}, ${optimiser}..."
        XLA_PYTHON_CLIENT_PREALLOCATE=false python hyperparameter_optimisation.py -c "configs/${dataset}.yaml" "configs/${optimiser}.yaml" configs/ASHA.yaml
        echo "Finished ${dataset}, ${optimiser}."
    done
done
echo "================================================================================"
echo "================================================================================"
echo "================================================================================"
