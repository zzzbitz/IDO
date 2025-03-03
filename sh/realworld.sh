#!/bin/bash

OUTPUT_DIR="" # change to your code path

config_files=("cifar100N.yaml" "webvision.yaml" "clothing1m.yaml")

datasets=("cifar100N" "webvision" "clothing1m")

CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoint/IDO_resnet50_realworld"

mkdir -p "${CHECKPOINT_DIR}"

for j in "${!config_files[@]}"; do

    LOG_FILE="${CHECKPOINT_DIR}/train_${datasets[$j]}_run${i}.log"
    
    echo "Running task with config: ${config_files[$j]}, dataset: ${datasets[$j]}, run: ${i}"
    
    nohup srun -p test_s2 --job-name=IDO --nodes=1 --gres=gpu:1 kernprof -l -v stage_1.py --cfg "config/stage_1/${config_files[$j]}" > "${LOG_FILE}" 2>&1
    nohup srun -p test_s2 --job-name=IDO --nodes=1 --gres=gpu:1 kernprof -l -v stage_2.py --cfg "config/stage_2/${config_files[$j]}" > "${LOG_FILE}" 2>&1
    echo "Task completed for config ${config_files[$j]}, dataset ${datasets[$j]}, run ${i}, log file: ${LOG_FILE}"
done
