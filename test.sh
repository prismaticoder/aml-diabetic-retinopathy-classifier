#!/bin/bash

# ─── Meta ─────────────────────────────────────────────────────
STUDENT_ID=6896375
STUDENT_NAME="Lawrence Attoh"
MODEL_NAME="mlp_mixer"
LEARNING_RATE="0.0003"
BATCH_SIZE="32"
WEIGHTS="DEFAULT"
LOG_DIR="logs"
CHECKPOINT_DIR="output/${MODEL_NAME}_lr${LEARNING_RATE}_bs${BATCH_SIZE}"
CONF_MATRIX_TITLE="Confusion Matrix for ${MODEL_NAME}"

# ─── Find Latest Checkpoint ──────────────────────────────────
CHECKPOINT_PATH=$(ls -t ${CHECKPOINT_DIR}/*.pth | head -n 1)

# ─── Run Test ────────────────────────────────────────────────
python3 test.py \
--model "$MODEL_NAME" \
--user "$STUDENT_NAME" \
--batch_size "$BATCH_SIZE" \
--learning_rate "$LEARNING_RATE" \
--csv_root_dir dataset \
--img_dir dataset/train \
--log_dir "$LOG_DIR" \
--evaluate_only \
--saved_checkpoint_path "$CHECKPOINT_PATH" \
--n_classes 5 \
--resized_img_weight 224 \
--resized_img_height 224 \
--seed 42 \
--weights "$WEIGHTS" \
--confusion_matrix_title "$CONF_MATRIX_TITLE"
