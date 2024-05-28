#!/bin/bash

datasets=(
    # 'uci_energy'
    # 'uci_protein'
    # 'fashion_mnist'
    # 'svhn'
    # 'cifar-10'
    'penn_treebank_gpt2_reset'
)

for dataset in ${datasets[@]}
do
    echo "================================================================================"
    echo "================================================================================"
    echo "================================================================================"
    echo "Starting ${dataset}, ${optimiser}..."
    #XLA_PYTHON_CLIENT_PREALLOCATE=false python hyperparameter_optimisation.py --SKIP_HPO -c "configs/${dataset}.yaml" "configs/AdamQLR_NoHPO_NoLRClipping_PTB.yaml" --log_root "/scratch/dir/ImprovingKFAC/runs/" -g "${dataset}_AdamQLR_NoHPO_NoLRClipping"
    XLA_PYTHON_CLIENT_PREALLOCATE=false python hyperparameter_optimisation.py --SKIP_HPO -c "configs/${dataset}.yaml" "configs/Adam.yaml" --log_root "/scratch/dir/ImprovingKFAC/runs/" -g "${dataset}_Adam_NoHPO"
    echo "Finished ${dataset}, ${optimiser}."
done
echo "================================================================================"
echo "================================================================================"
echo "================================================================================"
