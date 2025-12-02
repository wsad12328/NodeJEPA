#!/bin/bash

# Define seeds
seeds=(0 1000 2000 3000 4000 5000 6000 7000 8000 9000)

if [ -z "$1" ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi

dataset="$1"

# Run pretraining for each seed
for seed in "${seeds[@]}"; do
    echo "----------------------------------------------------------------"
    echo "Running pretraining for seed $seed..."
    echo "----------------------------------------------------------------"
    python pretrain_jepa.py --dataset $dataset --seed $seed
done

# Run downstream evaluation
echo "----------------------------------------------------------------"
echo "Running downstream evaluation..."
echo "----------------------------------------------------------------"
# Pass the seeds array to python script
seeds_str="${seeds[*]}"
python downstream.py --dataset $dataset --seeds $seeds_str
