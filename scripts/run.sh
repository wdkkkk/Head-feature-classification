#!/bin/bash
cd ..
# 定义学习率数组
learning_rates=(0.05 0.1)

# 定义模型数组
models=(''resnet18'' ''resnet34'' ''resnet50'')

# 遍历学习率和模型
for lr in "${learning_rates[@]}"
do
    for model in "${models[@]}"
    do
        echo "Training with learning rate: $lr and model: $model"
        CUDA_VISIBLE_DEVICES=7 python train.py --config /home/lff/wdk/wdk_self/AI_Project2/code/configs/resnet50.json -lr $lr --model $model
    done
done
