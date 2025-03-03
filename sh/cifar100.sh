#!/bin/bash

# Define the output directory
OUTPUT_DIR="" # change to your code path
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoint/IDO_resnet50_cifar100"

# Create the output and checkpoint directories if they do not exist
mkdir -p "${CHECKPOINT_DIR}"

# Define noise modes
noise_modes=("sym" "idn" "asym")

# Define noise ratios for each noise mode
declare -A noise_ratios
noise_ratios["sym"]="0.2 0.4 0.6"
noise_ratios["idn"]="0.4"
noise_ratios["asym"]="0.4"

# Iterate over each noise mode
for noise_mode in "${noise_modes[@]}"; do
    # Get the list of noise ratios for the current noise mode
    ratios=(${noise_ratios[$noise_mode]})
    for noise_ratio in "${ratios[@]}"; do
        # Define the log file path
        LOG_FILE="${CHECKPOINT_DIR}/train_${noise_mode}_r${noise_ratio}.log"

        # Print the current noise type and ratio
        echo "Training with noise mode: $noise_mode, noise ratio: $noise_ratio"
        nohup srun -p test_s2 --job-name=IDO --nodes=1 --gres=gpu:1 kernprof -l -v stage_1.py --cfg 'config/stage_1/cifar100.yaml' --noise_mode $noise_mode --noise_ratio $noise_ratio > "${LOG_FILE}" 2>&1
        nohup srun -p test_s2 --job-name=IDO --nodes=1 --gres=gpu:1 kernprof -l -v stage_2.py --cfg 'config/stage_2/cifar100.yaml' --noise_mode $noise_mode --noise_ratio $noise_ratio > "${LOG_FILE}" 2>&1
        echo "Training completed for noise mode ${noise_mode}, noise ratio ${noise_ratio}, log file: ${LOG_FILE}"

    done
done
