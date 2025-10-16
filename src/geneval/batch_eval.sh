#!/bin/bash

# ==================== Path Configuration ====================
BASE_ROOT="./geneval_results/unified_reward_results"
STEPS=20
OBJECT_DETECTOR_FOLDER="<OBJECT_DETECTOR_FOLDER>"  # Fill in the actual object detector model path
GPU_ID=0
# ==================== Path Configuration ====================
text_k=0.2
image_k=0.02
lr=0.03
reward_threshold=-1.0
benchmark="geneval" 
reward_model_type="unified_reward"
type="both"

# List of step markers
steps=("05" "10" "15" "20") 

# Iterate over each step value
for step in "${steps[@]}"; do
    # Directory and filenames for the current run
    BASE_DIR="${BASE_ROOT}/Janus-Pro-7B-${benchmark}-${reward_model_type}-${type}-text_k${text_k}-image_k${image_k}-steps${STEPS}-lr${lr}-reward_threshold${reward_threshold}"
    IMAGE_DIR="${BASE_DIR}/final_img_${step}"
    OUTFILE="${BASE_DIR}/results_text_k${text_k}-image_k${image_k}_s${step}_lr${lr}.jsonl"

    echo ">>> Starting evaluation: step=${step}"
    echo "BASE_DIR: ${BASE_DIR}"
    echo "OUTFILE: ${OUTFILE}"

    # Evaluation
    CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluation/evaluate_images.py \
        "${IMAGE_DIR}" \
        --outfile "${OUTFILE}" \
        --model-path "${OBJECT_DETECTOR_FOLDER}"

    # Summarize scores
    python evaluation/summary_scores.py "${OUTFILE}"
done

echo "✅ All evaluations completed!"
