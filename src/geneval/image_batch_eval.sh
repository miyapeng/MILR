#!/bin/bash

# ==================== 配置路径 ====================
BASE_ROOT="/media/raid/workspace/miyapeng/Multimodal-LatentSeek/src/geneval_results/3_results"
STEPS=15
#LR=0.01
OBJECT_DETECTOR_FOLDER="<OBJECT_DETECTOR_FOLDER>"  # 请填写实际对象检测模型路径
GPU_ID=2
# ==================== 配置路径 ====================
#text_k=0.2
#image_k=0.03
lr=0.01
reward_threshold=-0.1
# k 列表
image_k=("0.6" "0.7" "0.8" "0.9" "1.0")

# 遍历每个 k 值
for k in "${image_k[@]}"; do
    # 当前模型对应的目录与文件名
    BASE_DIR="${BASE_ROOT}/Janus-Pro-7B-geneval-geneval-image-image_k${k}-steps${STEPS}-lr${lr}-reward_threshold${reward_threshold}"
    IMAGE_DIR="${BASE_DIR}/final_img"
    OUTFILE="${BASE_DIR}/results_text_k${text_k}-image_k${image_k}_s${step}_lr${lr}.jsonl"

    echo ">>> 开始测试 step=${step}"
    echo "BASE_DIR: ${BASE_DIR}"
    echo "OUTFILE: ${OUTFILE}"

    # 评估
    CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluation/evaluate_images.py \
        "${IMAGE_DIR}" \
        --outfile "${OUTFILE}" \
        --model-path "${OBJECT_DETECTOR_FOLDER}"

    # 汇总
    python evaluation/summary_scores.py "${OUTFILE}"
done

echo "✅ 全部测试已完成！"
