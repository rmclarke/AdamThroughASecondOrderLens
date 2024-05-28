#!/bin/bash

datasets=(
    'uci_energy'
    'uci_protein'
    'fashion_mnist'
    'svhn'
    'cifar-10'
    # 'penn_treebank_gpt2_reset'
)

for dataset in ${datasets[@]}
do
    echo "================================================================================"
    echo "================================================================================"
    echo "================================================================================"
    echo "Starting ${dataset}, ${optimiser}..."
    XLA_PYTHON_CLIENT_PREALLOCATE=false python hyperparameter_optimisation.py --SKIP_HPO -c "configs/${dataset}.yaml" "configs/KFAC.yaml" --log_root "/scratch/dir/ImprovingKFAC/runs/" -g "${dataset}_KFAC_BigBatch_NoHPO" --batch_size=3200
    echo "Finished ${dataset}, ${optimiser}."
done
echo "================================================================================"
echo "================================================================================"
echo "================================================================================"
