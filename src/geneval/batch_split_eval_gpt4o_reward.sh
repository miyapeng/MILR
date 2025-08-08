#!/bin/bash

# ==================== Config ====================
BASE_ROOT="/media/raid/workspace/miyapeng/Multimodal-LatentSeek/src/geneval_results/gpt4o_reward"
STEPS=50
OBJECT_DETECTOR_FOLDER="<OBJECT_DETECTOR_FOLDER>"  # Fill in actual detector path
GPU_ID=1
text_k=0.2
image_k=0.02
lr=0.03
reward_threshold=-0.1
benchmark="geneval" 
reward_model_type="gpt4o"
type="both"

# Step list
# steps=("02" "04" "06" "08" "10" "12" "14" "16" "18" "20" "22" "24" "26" "28" "30")
steps=("05" "10" "15" "20" "25" "30" "35" "40" "45" "50")

# ==================== Preprocess: Split final_img by steps ====================
# BASE_DIR="${BASE_ROOT}/Janus-Pro-7B-${benchmark}-${reward_model_type}-${type}-text_k${text_k}-image_k${image_k}-steps${STEPS}-lr${lr}-reward_threshold${reward_threshold}"
if [ "$type" = "image" ]; then
        BASE_DIR="${BASE_ROOT}/Janus-Pro-7B-${benchmark}-${reward_model_type}-${type}-image_k${image_k}-steps${STEPS}-lr${lr}-reward_threshold${reward_threshold}"
    elif [ "$type" = "text" ]; then
        BASE_DIR="${BASE_ROOT}/Janus-Pro-7B-${benchmark}-${reward_model_type}-${type}-text_k${text_k}-steps${STEPS}-lr${lr}-reward_threshold${reward_threshold}"
    elif [ "$type" = "both" ]; then
        BASE_DIR="${BASE_ROOT}/Janus-Pro-7B-${benchmark}-${reward_model_type}-${type}-text_k${text_k}-image_k${image_k}-steps${STEPS}-lr${lr}-reward_threshold${reward_threshold}"
    else
        echo "[ERROR] Unknown type: $type"
        exit 1
    fi

# Convert steps to integer list (strip leading zero)
int_steps=()
for s in "${steps[@]}"; do
    int_steps+=($((10#$s)))  # 10# prevents octal interpretation
done

# Call the Python script to preprocess
echo ">>> Running step-splitting Python script on ${BASE_DIR}"
python process_geneval.py --parent_dir "$BASE_DIR" --steps "${int_steps[@]}"

# ==================== Evaluation Loop ====================
for step in "${steps[@]}"; do
    IMAGE_DIR="${BASE_DIR}/final_img_${step}"
    # OUTFILE="${BASE_DIR}/results_text_k${text_k}-image_k${image_k}_s${step}_lr${lr}.jsonl"

    if [ "$type" = "image" ]; then
        OUTFILE="${BASE_DIR}/results_both_image_k${image_k}_s${step}_lr${lr}.jsonl"
    elif [ "$type" = "text" ]; then
        OUTFILE="${BASE_DIR}/results_both_text_k${text_k}_s${step}_lr${lr}.jsonl"
    elif [ "$type" = "both" ]; then
        OUTFILE="${BASE_DIR}/results_both_text_k${text_k}_image_k${image_k}_s${step}_lr${lr}.jsonl"
    else
        echo "[ERROR] Unknown type: $type"
        exit 1
    fi

    echo ">>> Evaluating step=${step}"
    echo "IMAGE_DIR: ${IMAGE_DIR}"
    echo "OUTFILE: ${OUTFILE}"

    # Evaluation
    CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluation/evaluate_images.py \
        "${IMAGE_DIR}" \
        --outfile "${OUTFILE}" \
        --model-path "${OBJECT_DETECTOR_FOLDER}"

    # Summary
    python evaluation/summary_scores.py "${OUTFILE}"
done

echo "✅ All evaluations completed!"

#!/bin/bash

# # ========== 基础配置 ==========
# BASE_ROOT="/media/raid/workspace/miyapeng/Multimodal-LatentSeek/src/geneval_results/parameters_image_k_results_seed43"
# STEPS=30
# OBJECT_DETECTOR_FOLDER="<OBJECT_DETECTOR_FOLDER>"  # ⛳ 填写模型路径
# GPU_LIST=(0 1 2 3 4 5 6 7)  # 你的可用 GPU
# MAX_GPU=${#GPU_LIST[@]}

# text_k=0.2
# lr=0.03
# reward_threshold=-0.1
# benchmark="geneval"
# reward_model_type="geneval"
# type="image"
# steps=("05" "10" "15" "20" "25" "30")

# image_k_list=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2)

# # ========== 遍历 text_k 并发执行 ==========
# job_idx=0

# for image_k in "${image_k_list[@]}"; do
#     gpu_id=${GPU_LIST[$((job_idx % MAX_GPU))]}
#     ((job_idx++))

#     (
#         echo ">>> [image_k=${image_k}] 分配到 GPU ${gpu_id} 上开始评测"

#         # BASE_DIR="${BASE_ROOT}/Janus-Pro-7B-${benchmark}-${reward_model_type}-${type}-text_k${text_k}-image_k${image_k}-steps${STEPS}-lr${lr}-reward_threshold${reward_threshold}"
#         BASE_DIR="${BASE_ROOT}/Janus-Pro-7B-${benchmark}-${reward_model_type}-${type}-image_k${image_k}-steps${STEPS}-lr${lr}-reward_threshold${reward_threshold}"

#         # 步骤 1：预处理（图像分步）
#         int_steps=()
#         for s in "${steps[@]}"; do
#             int_steps+=($((10#$s)))
#         done

#         echo ">>> [image_k=${image_k}] 预处理图像分步..."
#         python process_geneval.py --parent_dir "$BASE_DIR" --steps "${int_steps[@]}"

#         # 步骤 2：按步数顺序评测（在同一 GPU）
#         for step in "${steps[@]}"; do
#             IMAGE_DIR="${BASE_DIR}/final_img_${step}"
#             OUTFILE="${BASE_DIR}/results_image_k${image_k}_s${step}_lr${lr}.jsonl"

#             echo ">>> [image_k=${image_k}] step=${step} 评测中 (GPU ${gpu_id})"
#             CUDA_VISIBLE_DEVICES=${gpu_id} python evaluation/evaluate_images.py \
#                 "${IMAGE_DIR}" \
#                 --outfile "${OUTFILE}" \
#                 --model-path "${OBJECT_DETECTOR_FOLDER}"

#             python evaluation/summary_scores.py "${OUTFILE}"
#         done

#         echo "✅ [image_k=${image_k}] 全部步骤评测完成 (GPU ${gpu_id})"
#     ) &  

#     sleep 1
# done

# wait
# echo "✅ 所有 image_k 并发任务全部完成！"

# # ========== 基础配置 ==========
# BASE_ROOT="/media/raid/workspace/miyapeng/Multimodal-LatentSeek/src/geneval_results/parameters_text_k_results_seed43"
# STEPS=30
# OBJECT_DETECTOR_FOLDER="<OBJECT_DETECTOR_FOLDER>"  # ⛳ 填写模型路径
# GPU_LIST=(0 1 2 3 4 5 6 7)  # 你的可用 GPU
# MAX_GPU=${#GPU_LIST[@]}

# image_k=0.02
# lr=0.03
# reward_threshold=-0.1
# benchmark="geneval"
# reward_model_type="geneval"
# type="text"
# steps=("05" "10" "15" "20" "25" "30")

# text_k_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# # ========== 遍历 text_k 并发执行 ==========
# job_idx=0

# for text_k in "${text_k_list[@]}"; do
#     gpu_id=${GPU_LIST[$((job_idx % MAX_GPU))]}
#     ((job_idx++))

#     (
#         echo ">>> [text_k=${text_k}] 分配到 GPU ${gpu_id} 上开始评测"

#         # BASE_DIR="${BASE_ROOT}/Janus-Pro-7B-${benchmark}-${reward_model_type}-${type}-text_k${text_k}-image_k${image_k}-steps${STEPS}-lr${lr}-reward_threshold${reward_threshold}"
#         BASE_DIR="${BASE_ROOT}/Janus-Pro-7B-${benchmark}-${reward_model_type}-${type}-text_k${text_k}-steps${STEPS}-lr${lr}-reward_threshold${reward_threshold}"

#         # 步骤 1：预处理（图像分步）
#         int_steps=()
#         for s in "${steps[@]}"; do
#             int_steps+=($((10#$s)))
#         done

#         echo ">>> [text_k=${text_k}] 预处理图像分步..."
#         python process_geneval.py --parent_dir "$BASE_DIR" --steps "${int_steps[@]}"

#         # 步骤 2：按步数顺序评测（在同一 GPU）
#         for step in "${steps[@]}"; do
#             IMAGE_DIR="${BASE_DIR}/final_img_${step}"
#             OUTFILE="${BASE_DIR}/results_text_k${text_k}_s${step}_lr${lr}.jsonl"

#             echo ">>> [text_k=${text_k}] step=${step} 评测中 (GPU ${gpu_id})"
#             CUDA_VISIBLE_DEVICES=${gpu_id} python evaluation/evaluate_images.py \
#                 "${IMAGE_DIR}" \
#                 --outfile "${OUTFILE}" \
#                 --model-path "${OBJECT_DETECTOR_FOLDER}"

#             python evaluation/summary_scores.py "${OUTFILE}"
#         done

#         echo "✅ [text_k=${text_k}] 全部步骤评测完成 (GPU ${gpu_id})"
#     ) &  

#     sleep 1
# done

# wait
# echo "✅ 所有 text_k 并发任务全部完成！"
