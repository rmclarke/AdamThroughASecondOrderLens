#!/bin/bash

# BEFORE RUNNING:
# Edit hyperparameter_optimisation.py `if __name__ == '__main__'` block to execute run_all_ablations()

optimisers=(
    'SGD'
    'AdamQLR_Damped_NoLRClipping'
    'Adam'
    'KFAC'
    'SGDmwd'
    'AdamQLR_Damped'
    'AdamQLR_Undamped_NoLRClipping'
    'AdamQLR_Damped_Hessian_NoLRClipping'
)

datasets=(
    'uci_energy'
    'uci_protein'
    'fashion_mnist'
    'svhn'
    'cifar-10'
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
        XLA_PYTHON_CLIENT_PREALLOCATE=false python hyperparameter_optimisation.py -c "configs/${dataset}.yaml" "configs/${optimiser}.yaml" configs/ASHA_local.yaml
        echo "Finished ${dataset}, ${optimiser}."
    done
done
echo "================================================================================"
echo "================================================================================"
echo "================================================================================"
