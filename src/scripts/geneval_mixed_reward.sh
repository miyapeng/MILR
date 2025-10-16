#!/bin/bash

PATH_TO_DATA="prompts/geneval/evaluation_metadata.jsonl"
PATH_TO_MODEL="deepseek-ai/Janus-Pro-7B"
output_dir="./geneval_results/mixed_reward_results"
optimize_mode="both"  # or "image"
reward_model_type="mixed_reward"
reward_threshold=-0.2
text_k=0.2 
image_k=0.02 
lr=0.03
max_text_steps=20
max_image_steps=20
max_both_steps=20

# === set log file name ===
if [ "$optimize_mode" = "text" ]; then
    LOG_FILE="$output_dir/${optimize_mode}_tk${text_k}_lr${lr}_ts${max_text_steps}_threshold${reward_threshold}.txt"
elif [ "$optimize_mode" = "image" ]; then
    LOG_FILE="$output_dir/${optimize_mode}_ik${image_k}_lr${lr}_is${max_image_steps}_threshold${reward_threshold}.txt"
else
    LOG_FILE="$output_dir/${optimize_mode}_tk${text_k}_ik${image_k}_lr${lr}_bs${max_both_steps}_threshold${reward_threshold}.txt"
fi

# === start training script ===
CUDA_VISIBLE_DEVICES=0 python main_janus.py \
    --dataset "$PATH_TO_DATA" \
    --model_name_or_path "$PATH_TO_MODEL" \
    --output_dir "$output_dir" \
    --optimize_mode "$optimize_mode" \
    --reward_model_type "$reward_model_type" \
    --lr "$lr" \
    --text_k "$text_k" \
    --image_k "$image_k" \
    --max_text_steps "$max_text_steps" \
    --max_image_steps "$max_image_steps" \
    --max_both_steps "$max_both_steps" \
    --device "cuda" \
    --reward_threshold "$reward_threshold" \
    > "$LOG_FILE" 2>&1 &
