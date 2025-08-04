#!/bin/bash

PATH_TO_DATA="prompts/geneval/evaluation_metadata.jsonl"
PATH_TO_MODEL="deepseek-ai/Janus-Pro-7B"
output_dir="./geneval_results/3_results"
optimize_mode="image"  # or "image"
reward_model_type="geneval" 
image_k=0.02 
lr=0.03
max_image_steps=30

# === 设置日志文件名 ===
if [ "$optimize_mode" = "text" ]; then
    LOG_FILE="$output_dir/${optimize_mode}_tk${text_k}_lr${lr}_ts${max_text_steps}.txt"
elif [ "$optimize_mode" = "image" ]; then
    LOG_FILE="$output_dir/${optimize_mode}_ik${image_k}_lr${lr}_is${max_image_steps}.txt"
else
    LOG_FILE="$output_dir/${optimize_mode}_tk${text_k}_ik${image_k}_lr${lr}_bs${max_both_steps}.txt"
fi

# === 启动训练脚本 ===
CUDA_VISIBLE_DEVICES=0 python main_janus.py \
    --dataset "$PATH_TO_DATA" \
    --model_name_or_path "$PATH_TO_MODEL" \
    --output_dir "$output_dir" \
    --optimize_mode "$optimize_mode" \
    --reward_model_type "$reward_model_type" \
    --lr "$lr" \
    --image_k "$image_k" \
    --max_image_steps "$max_image_steps" \
    --device "cuda" \
    > "$LOG_FILE" 2>&1 &
