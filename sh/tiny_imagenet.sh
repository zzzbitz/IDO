#!/bin/bash

# Define the output directory
OUTPUT_DIR="" # change to your code path
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoint/IDO_resnet50_tinyimagenet_cifar100"

mkdir -p "${CHECKPOINT_DIR}"

noise_modes=("sym" "idn")

declare -A noise_ratios
# noise_ratios["sym"]="0.2 0.5"
noise_ratios["idn"]="0.4"

for noise_mode in "${noise_modes[@]}"; do

    ratios=(${noise_ratios[$noise_mode]})
    for noise_ratio in "${ratios[@]}"; do

        LOG_FILE="${CHECKPOINT_DIR}/train_${noise_mode}_r${noise_ratio}.log"

        echo "Training with noise mode: $noise_mode, noise ratio: $noise_ratio"
        nohup srun -p test_s2 --job-name=IDO --nodes=1 --gres=gpu:1 kernprof -l -v stage_1.py --cfg 'config/stage_1/tiny_imagenet.yaml' --noise_mode $noise_mode --noise_ratio $noise_ratio > "${LOG_FILE}" 2>&1
        nohup srun -p test_s2 --job-name=IDO --nodes=1 --gres=gpu:1 kernprof -l -v stage_2.py --cfg 'config/stage_2/tiny_imagenet.yaml' --noise_mode $noise_mode --noise_ratio $noise_ratio > "${LOG_FILE}" 2>&1
        echo "Training completed for noise mode ${noise_mode}, noise ratio ${noise_ratio}, log file: ${LOG_FILE}"  

    done
done
