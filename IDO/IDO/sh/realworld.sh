#!/bin/bash

# 定义输出目录
OUTPUT_DIR="/mnt/lustre/zk/ICML2025/IDO" # change to your code path

# 定义配置文件列表
config_files=("cifar100N.yaml" "webvision.yaml" "clothing1m.yaml")

# 定义数据集名称列表（用于日志文件命名）
datasets=("cifar100N" "webvision" "clothing1m")

# 定义检查点目录（统一为 ELR_resnet50_realworld）
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoint/IDO_resnet50_realworld"

# 创建检查点目录（如果不存在）
mkdir -p "${CHECKPOINT_DIR}"



# 遍历每个配置文件和数据集
for j in "${!config_files[@]}"; do
    # 定义日志文件路径
    LOG_FILE="${CHECKPOINT_DIR}/train_${datasets[$j]}_run${i}.log"
    
    # 打印当前任务信息
    echo "Running task with config: ${config_files[$j]}, dataset: ${datasets[$j]}, run: ${i}"
    
    # 执行任务并将输出重定向到日志文件
    nohup srun -p test_s2 --job-name=IDO --nodes=1 --gres=gpu:1 kernprof -l -v stage_1.py --cfg "config/stage_1/${config_files[$j]}" > "${LOG_FILE}" 2>&1
    nohup srun -p test_s2 --job-name=IDO --nodes=1 --gres=gpu:1 kernprof -l -v stage_2.py --cfg "config/stage_2/${config_files[$j]}" > "${LOG_FILE}" 2>&1
    echo "Task completed for config ${config_files[$j]}, dataset ${datasets[$j]}, run ${i}, log file: ${LOG_FILE}"
done
