#!/bin/bash

optimisers=(
    'SGD'
    'SGDmwd'
    'Adam'
    'KFAC'
    'AdamQLR_Damped_NoLRClipping'
)

datasets=(
    'uci_energy'
    'uci_protein'
    'fashion_mnist'
    'svhn'
    'cifar-10_long'
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
        XLA_PYTHON_CLIENT_PREALLOCATE=false python hyperparameter_optimisation.py -c "configs/${dataset}.yaml" "configs/${optimiser}.yaml" configs/ASHA_time_training.yaml --log_root "/scratch/dir/ImprovingKFAC/runs/ASHA_Time_Training"
        echo "Finished ${dataset}, ${optimiser}."
    done
done
echo "================================================================================"
echo "================================================================================"
echo "================================================================================"
