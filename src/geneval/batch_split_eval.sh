#!/bin/bash

# ==================== Config ====================
BASE_ROOT="./geneval_results/Janus-Pro-1B"
STEPS=20
OBJECT_DETECTOR_FOLDER="<OBJECT_DETECTOR_FOLDER>"  # Fill in actual detector path
GPU_ID=0
text_k=0.2
image_k=0.02
lr=0.03
reward_threshold=-0.1
benchmark="geneval" 
reward_model_type="geneval"
type="image"

# Step list
steps=("02" "04" "06" "08" "10" "12" "14" "16" "18" "20")

# ==================== Preprocess: Split final_img by steps ====================
# BASE_DIR="${BASE_ROOT}/Janus-Pro-7B-${benchmark}-${reward_model_type}-${type}-text_k${text_k}-image_k${image_k}-steps${STEPS}-lr${lr}-reward_threshold${reward_threshold}"
if [ "$type" = "image" ]; then
        BASE_DIR="${BASE_ROOT}/Janus-Pro-1B-${benchmark}-${reward_model_type}-${type}-image_k${image_k}-steps${STEPS}-lr${lr}-reward_threshold${reward_threshold}"
    elif [ "$type" = "text" ]; then
        BASE_DIR="${BASE_ROOT}/Janus-Pro-1B-${benchmark}-${reward_model_type}-${type}-text_k${text_k}-steps${STEPS}-lr${lr}-reward_threshold${reward_threshold}"
    elif [ "$type" = "both" ]; then
        BASE_DIR="${BASE_ROOT}/Janus-Pro-1B-${benchmark}-${reward_model_type}-${type}-text_k${text_k}-image_k${image_k}-steps${STEPS}-lr${lr}-reward_threshold${reward_threshold}"
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