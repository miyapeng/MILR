#!/bin/bash

# ==================== 配置路径 ====================
OBJECT_DETECTOR_FOLDER="<OBJECT_DETECTOR_FOLDER>"  # 请填写实际对象检测模型路径
GPU_ID=1
# ==================== 配置路径 ====================
IMAGE_DIR="/fs-computility/ai-shen/fanyuyu/latentseek/Multimodal-LatentSeek/src/geneval_results/Time_Analysis/Janus-Pro-7B-geneval-geneval-text_image-text_k0.2-image_k0.02-steps30-lr0.03-reward_threshold-0.1/final_img"
OUTFILE="/fs-computility/ai-shen/fanyuyu/latentseek/Multimodal-LatentSeek/src/geneval_results/Time_Analysis/Janus-Pro-7B-geneval-geneval-text_image-text_k0.2-image_k0.02-steps30-lr0.03-reward_threshold-0.1/results.jsonl"

echo ">>> 开始测试"

# 评估
CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluation/evaluate_images.py \
    "${IMAGE_DIR}" \
    --outfile "${OUTFILE}" \
    --model-path "${OBJECT_DETECTOR_FOLDER}"

# 汇总
python evaluation/summary_scores.py "${OUTFILE}"

