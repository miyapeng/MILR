# #!/bin/bash

# # ==================== 配置路径 ====================
# BASE_ROOT="/media/raid/workspace/miyapeng/Multimodal-LatentSeek/src/geneval_results/parameters_text_k_results"
# OBJECT_DETECTOR_FOLDER="<OBJECT_DETECTOR_FOLDER>"  # fill the real detection path
# STEPS=30
# LR=0.03
# GPU_ID=2
# text_k=1.0
# image_k=0.01
# reward_model_type="geneval"
# reward_threshold=-0.1
# type="both"  # 可选: "image" | "text" | "both"
# benchmark="geneval" 
# # ==================== 配置路径 ====================

# # 根据 type 设置 BASE_DIR 和 OUTFILE
# if [ "$type" = "image" ]; then
#     BASE_DIR="${BASE_ROOT}/Janus-Pro-7B-${benchmark}-${reward_model_type}-${type}-image_k${image_k}-steps${STEPS}-lr${LR}-reward_threshold${reward_threshold}"
#     OUTFILE="${BASE_DIR}/results_both_image_k${image_k}_s${STEPS}_lr${LR}.jsonl"
# elif [ "$type" = "text" ]; then
#     BASE_DIR="${BASE_ROOT}/Janus-Pro-7B-${benchmark}-${reward_model_type}-${type}-text_k${text_k}-steps${STEPS}-lr${LR}-reward_threshold${reward_threshold}"
#     OUTFILE="${BASE_DIR}/results_both_text_k${text_k}_s${STEPS}_lr${LR}_ori.jsonl"
# elif [ "$type" = "both" ]; then
#     BASE_DIR="${BASE_ROOT}/Janus-Pro-7B-${benchmark}-${reward_model_type}-${type}-text_k${text_k}-image_k${image_k}-steps${STEPS}-lr${LR}-reward_threshold${reward_threshold}"
#     OUTFILE="${BASE_DIR}/results_both_text_k${text_k}_image_k${image_k}_s${STEPS}_lr${LR}_ori.jsonl"
# else
#     echo "[ERROR] Unknown type: $type"
#     exit 1
# fi

# # IMAGE_DIR="${BASE_DIR}/final_img"
# IMAGE_DIR="${BASE_DIR}/ori_img"

# echo ">>> 开始测试"
# echo "BASE_DIR: ${BASE_DIR}"
# echo "OUTFILE: ${OUTFILE}"

# # 评估
# CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluation/evaluate_images.py \
#     "${IMAGE_DIR}" \
#     --outfile "${OUTFILE}" \
#     --model-path "${OBJECT_DETECTOR_FOLDER}"

# # 汇总
# python evaluation/summary_scores.py "${OUTFILE}"

#!/bin/bash

# # ==================== 通用配置 ====================
# BASE_ROOT="/media/raid/workspace/miyapeng/Multimodal-LatentSeek/src/geneval_results/parameters_image_k_results_seed42"
# OBJECT_DETECTOR_FOLDER="<OBJECT_DETECTOR_FOLDER>"  # ⛳ 填入检测模型路径
# STEPS=30
# LR=0.03
# text_k=0.2
# reward_model_type="geneval"
# reward_threshold=-0.1
# type="both"  # 可选: "image" | "text" | "both"
# benchmark="geneval"
# GPU_LIST=(0 1 2 5 6 7)  # 你的 GPU 编号列表
# MAX_GPU=${#GPU_LIST[@]}

# # 所有需要测试的 text_k
# image_k_list=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4)

# # ==================== 启动任务 ====================
# job_idx=0

# for image_k in "${image_k_list[@]}"; do
#     gpu_id=${GPU_LIST[$((job_idx % MAX_GPU))]}
#     ((job_idx++))

#     (
#         # 计算路径
#         if [ "$type" = "image" ]; then
#             BASE_DIR="${BASE_ROOT}/Janus-Pro-7B-${benchmark}-${reward_model_type}-${type}-image_k${image_k}-steps${STEPS}-lr${LR}-reward_threshold${reward_threshold}"
#             OUTFILE="${BASE_DIR}/results_image_image_k${image_k}_s${STEPS}_lr${LR}.jsonl"
#         elif [ "$type" = "text" ]; then
#             BASE_DIR="${BASE_ROOT}/Janus-Pro-7B-${benchmark}-${reward_model_type}-${type}-text_k${text_k}-steps${STEPS}-lr${LR}-reward_threshold${reward_threshold}"
#             OUTFILE="${BASE_DIR}/results_text_text_k${text_k}_s${STEPS}_lr${LR}_ori.jsonl"
#         elif [ "$type" = "both" ]; then
#             BASE_DIR="${BASE_ROOT}/Janus-Pro-7B-${benchmark}-${reward_model_type}-${type}-text_k${text_k}-image_k${image_k}-steps${STEPS}-lr${LR}-reward_threshold${reward_threshold}"
#             OUTFILE="${BASE_DIR}/results_both_text_k${text_k}_image_k${image_k}_s${STEPS}_lr${LR}_ori.jsonl"
#         else
#             echo "[ERROR] Unknown type: $type"
#             exit 1
#         fi

#         IMAGE_DIR="${BASE_DIR}/ori_img"

#         echo ">>> [image_k=${image_k}] Starting on GPU ${gpu_id}"
#         echo "BASE_DIR: ${BASE_DIR}"
#         echo "OUTFILE: ${OUTFILE}"

#         # 评估
#         CUDA_VISIBLE_DEVICES=${gpu_id} python evaluation/evaluate_images.py \
#             "${IMAGE_DIR}" \
#             --outfile "${OUTFILE}" \
#             --model-path "${OBJECT_DETECTOR_FOLDER}"

#         # 汇总
#         python evaluation/summary_scores.py "${OUTFILE}"

#         echo "✅ [image_k=${image_k}] Done on GPU ${gpu_id}"
#     ) &
#     sleep 1  # 避免并发启动瞬间资源抢占
# done

# wait
# echo "✅ 所有 image_k 的评估已完成"

# ==================== 通用配置 ====================
BASE_ROOT="/media/raid/workspace/miyapeng/Multimodal-LatentSeek/src/geneval_results/parameters_text_k_results_seed43"
OBJECT_DETECTOR_FOLDER="<OBJECT_DETECTOR_FOLDER>"  # ⛳ 填入检测模型路径
STEPS=30
LR=0.03
image_k=0.02
reward_model_type="geneval"
reward_threshold=-0.1
type="text"  # 可选: "image" | "text" | "both"
benchmark="geneval"
GPU_LIST=(0 1 2 3 4 5 6 7)  # 你的 GPU 编号列表
MAX_GPU=${#GPU_LIST[@]}

# 所有需要测试的 text_k
text_k_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# ==================== 启动任务 ====================
job_idx=0

for text_k in "${text_k_list[@]}"; do
    gpu_id=${GPU_LIST[$((job_idx % MAX_GPU))]}
    ((job_idx++))

    (
        # 计算路径
        if [ "$type" = "image" ]; then
            BASE_DIR="${BASE_ROOT}/Janus-Pro-7B-${benchmark}-${reward_model_type}-${type}-image_k${image_k}-steps${STEPS}-lr${LR}-reward_threshold${reward_threshold}"
            OUTFILE="${BASE_DIR}/results_image_image_k${image_k}_s${STEPS}_lr${LR}.jsonl"
        elif [ "$type" = "text" ]; then
            BASE_DIR="${BASE_ROOT}/Janus-Pro-7B-${benchmark}-${reward_model_type}-${type}-text_k${text_k}-steps${STEPS}-lr${LR}-reward_threshold${reward_threshold}"
            OUTFILE="${BASE_DIR}/results_text_text_k${text_k}_s${STEPS}_lr${LR}_ori.jsonl"
        elif [ "$type" = "both" ]; then
            BASE_DIR="${BASE_ROOT}/Janus-Pro-7B-${benchmark}-${reward_model_type}-${type}-text_k${text_k}-image_k${image_k}-steps${STEPS}-lr${LR}-reward_threshold${reward_threshold}"
            OUTFILE="${BASE_DIR}/results_both_text_k${text_k}_image_k${image_k}_s${STEPS}_lr${LR}_ori.jsonl"
        else
            echo "[ERROR] Unknown type: $type"
            exit 1
        fi

        IMAGE_DIR="${BASE_DIR}/ori_img"

        echo ">>> [text_k=${text_k}] Starting on GPU ${gpu_id}"
        echo "BASE_DIR: ${BASE_DIR}"
        echo "OUTFILE: ${OUTFILE}"

        # 评估
        CUDA_VISIBLE_DEVICES=${gpu_id} python evaluation/evaluate_images.py \
            "${IMAGE_DIR}" \
            --outfile "${OUTFILE}" \
            --model-path "${OBJECT_DETECTOR_FOLDER}"

        # 汇总
        python evaluation/summary_scores.py "${OUTFILE}"

        echo "✅ [text_k=${text_k}] Done on GPU ${gpu_id}"
    ) &
    sleep 1  # 避免并发启动瞬间资源抢占
done

wait
echo "✅ 所有 text_k 的评估已完成"