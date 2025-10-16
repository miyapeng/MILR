#!/bin/bash

# ==================== Path Configuration ====================
BASE_ROOT="./geneval_results/long_results"
STEPS=20
#LR=0.01
OBJECT_DETECTOR_FOLDER="<OBJECT_DETECTOR_FOLDER>"  # Fill in the actual object detector model path
GPU_ID=6
# ==================== Path Configuration ====================
text_k=0.2
image_k=0.02
lr=0.03
reward_threshold=-0.1
# List of step suffixes
steps=("05" "10" "15" "20")

# Iterate over each step value
for step in "${steps[@]}"; do
    # Directory and filenames for the current model/run
    BASE_DIR="${BASE_ROOT}/Janus-Pro-7B-geneval-geneval-image-image_k${image_k}-steps${STEPS}-lr${lr}-reward_threshold${reward_threshold}"
    IMAGE_DIR="${BASE_DIR}/final_img_${step}"
    OUTFILE="${BASE_DIR}/results_image_k${image_k}_s${step}_lr${lr}.jsonl"

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
