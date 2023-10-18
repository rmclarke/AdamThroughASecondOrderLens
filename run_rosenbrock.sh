#!/bin/bash

optimisers=(
    'SGD'
    'SGDmwd'
    'Adam'
    'AdamQLR_Damped_Hessian'
)

datasets=(
    'rosenbrock'
)

for dataset in ${datasets[@]}
do
    for optimiser in ${optimisers[@]}
    do
        echo "================================================================================"
        echo "================================================================================"
        echo "================================================================================"
        echo "Starting ${dataset}, ${optimiser}..."
        XLA_PYTHON_CLIENT_PREALLOCATE=false python hyperparameter_optimisation.py -c "configs/${dataset}.yaml" "configs/${optimiser}.yaml" "configs/ASHA_local.yaml" \
            --tuning_metric="training_loss" --log_root="/scratch/dir/ImprovingKFAC/runs" --run_group_name="${dataset}" --run_name="${optimiser}"
        echo "Finished ${dataset}, ${optimiser}."
    done
done
echo "================================================================================"
echo "================================================================================"
echo "================================================================================"
