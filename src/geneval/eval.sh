#!/bin/bash

# ==================== 配置路径 ====================
BASE_ROOT="/media/raid/workspace/miyapeng/Multimodal-LatentSeek/src/geneval_results/mixed_reward1_results"
STEPS=10
LR=0.01
OBJECT_DETECTOR_FOLDER="<OBJECT_DETECTOR_FOLDER>"  # 请填写实际对象检测模型路径
GPU_ID=1
text_k=0.1
image_k=0.01
reward_model_type="mixed_reward"
reward_threshold=-0.1
type="text"
benchmark="geneval" 
# ==================== 配置路径 ====================
# 当前模型对应的目录与文件名
#BASE_DIR="${BASE_ROOT}/Janus-Pro-7B-${benchmark}-${reward_model_type}-${type}-text_k${text_k}-image_k${image_k}-steps${STEPS}-lr${LR}-reward_threshold${reward_threshold}"
#BASE_DIR="${BASE_ROOT}/Janus-Pro-7B-${benchmark}-${reward_model_type}-${type}-image_k${image_k}-steps${STEPS}-lr${LR}-reward_threshold${reward_threshold}"
BASE_DIR="${BASE_ROOT}/Janus-Pro-7B-${benchmark}-${reward_model_type}-${type}-text_k${text_k}-steps${STEPS}-lr${LR}-reward_threshold${reward_threshold}"
IMAGE_DIR="${BASE_DIR}/final_img"
#OUTFILE="${BASE_DIR}/results_both_text_k${text_k}_image_k${image_k}_s${STEPS}_lr${LR}_threshold${reward_threshold}.jsonl"
#OUTFILE="${BASE_DIR}/results_both_image_k${image_k}_s${STEPS}_lr${LR}_threshold${reward_threshold}.jsonl"
OUTFILE="${BASE_DIR}/results_both_text_k${text_k}_s${STEPS}_lr${LR}_threshold${reward_threshold}.jsonl"

echo ">>> 开始测试"
echo "BASE_DIR: ${BASE_DIR}"
echo "OUTFILE: ${OUTFILE}"

# 评估
CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluation/evaluate_images.py \
    "${IMAGE_DIR}" \
    --outfile "${OUTFILE}" \
    --model-path "${OBJECT_DETECTOR_FOLDER}"

# 汇总
python evaluation/summary_scores.py "${OUTFILE}"

