#!/bin/bash

# ==================== Path Configuration ====================
OBJECT_DETECTOR_FOLDER="<OBJECT_DETECTOR_FOLDER>"  # Fill in the actual object detector model path
GPU_ID=0
# ==================== Path Configuration ====================
IMAGE_DIR=""
OUTFILE=""

echo ">>> Start testing"

# Evaluation
CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluation/evaluate_images.py \
    "${IMAGE_DIR}" \
    --outfile "${OUTFILE}" \
    --model-path "${OBJECT_DETECTOR_FOLDER}"

# Summarize scores
python evaluation/summary_scores.py "${OUTFILE}"
