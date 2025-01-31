#!/bin/bash

# Define the output directory
OUTPUT_DIR="/mnt/lustre/zk/ICML2025/IDO" # change to your code path
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoint/IDO_resnet50_tinyimagenet_cifar100"

# 创建输出目录和检查点目录（如果不存在）
mkdir -p "${CHECKPOINT_DIR}"

# 定义噪声模式
noise_modes=("sym" "idn")

# 定义每个噪声模式对应的噪声比值
declare -A noise_ratios
# noise_ratios["sym"]="0.2 0.5"
noise_ratios["idn"]="0.4"
# 遍历每个噪声模式
for noise_mode in "${noise_modes[@]}"; do
    # 获取当前噪声模式对应的噪声比值列表
    ratios=(${noise_ratios[$noise_mode]})
    for noise_ratio in "${ratios[@]}"; do
        # 定义日志文件路径
        LOG_FILE="${CHECKPOINT_DIR}/train_${noise_mode}_r${noise_ratio}.log"
        
        # 打印当前噪声类型和比例
        echo "Training with noise mode: $noise_mode, noise ratio: $noise_ratio"
        nohup srun -p test_s2 --job-name=IDO --nodes=1 --gres=gpu:1 kernprof -l -v stage_1.py --cfg 'config/stage_1/tiny_imagenet.yaml' --noise_mode $noise_mode --noise_ratio $noise_ratio > "${LOG_FILE}" 2>&1
        nohup srun -p test_s2 --job-name=IDO --nodes=1 --gres=gpu:1 kernprof -l -v stage_2.py --cfg 'config/stage_2/tiny_imagenet.yaml' --noise_mode $noise_mode --noise_ratio $noise_ratio > "${LOG_FILE}" 2>&1
        echo "Training completed for noise mode ${noise_mode}, noise ratio ${noise_ratio}, log file: ${LOG_FILE}"  

    done
done
