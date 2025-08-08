#!/bin/bash

# ==================== 配置路径 ====================
OBJECT_DETECTOR_FOLDER="<OBJECT_DETECTOR_FOLDER>"  # 请填写实际对象检测模型路径
GPU_ID=1
# ==================== 配置路径 ====================
IMAGE_DIR="/media/raid/workspace/miyapeng/Multimodal-LatentSeek/src/geneval_results/janus-pro-BoN-10"
OUTFILE="/media/raid/workspace/miyapeng/Multimodal-LatentSeek/src/geneval_results/janus-pro-BoN-10/results.jsonl"

echo ">>> 开始测试"

# 评估
CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluation/evaluate_images.py \
    "${IMAGE_DIR}" \
    --outfile "${OUTFILE}" \
    --model-path "${OBJECT_DETECTOR_FOLDER}"

# 汇总
python evaluation/summary_scores.py "${OUTFILE}"

