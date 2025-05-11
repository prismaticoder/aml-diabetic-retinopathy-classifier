#!/bin/bash

# ─── Meta Info ──────────────────────────────────────────────────────────────
STUDENT_NAME="Lawrence Attoh"
MODEL_NAME="mlp_mixer_v2_batchnorm"
LEARNING_RATE="0.0001"
BATCH_SIZE="32"
WEIGHTS="DEFAULT"
LOG_DIR="logs"
CSV_DIR="dataset"
IMG_DIR="dataset/test"
CONF_MATRIX_TITLE="Confusion Matrix for ${MODEL_NAME}"
CHECKPOINT_DIR="output/${MODEL_NAME}_lr${LEARNING_RATE}_bs${BATCH_SIZE}"

# ─── Auto-Find Most Recent Checkpoint ───────────────────────────────────────
CHECKPOINT_PATH=$(ls -t ${CHECKPOINT_DIR}/*.pth | head -n 1)

# ─── Run Test ───────────────────────────────────────────────────────────────
python3 test.py \
--model "$MODEL_NAME" \
--user "$STUDENT_NAME" \
--batch_size "$BATCH_SIZE" \
--learning_rate "$LEARNING_RATE" \
--csv_root_dir "$CSV_DIR" \
--img_dir "$IMG_DIR" \
--train_datacsv "$CSV_DIR/train.csv" \
--val_datacsv "$CSV_DIR/val.csv" \
--test_datacsv "$CSV_DIR/test.csv" \
--saved_checkpoint_path "$CHECKPOINT_PATH" \
--log_dir "$LOG_DIR" \
--weights "$WEIGHTS" \
--confusion_matrix_title "$CONF_MATRIX_TITLE" \
--n_classes 5 \
--resized_img_weight 224 \
--resized_img_height 224 \
--seed 42 \
--evaluate_only
