#!/bin/bash

# ─── Meta Info ────────────────────────────────────────────────
STUDENT_NAME="Lawrence Attoh"
MODEL_NAME="mlp_mixer_v2_addblock"
LEARNING_RATE="0.0005"
BATCH_SIZE="128"
PATCH_SIZE="16"
N_CLASSES=5
RESIZE=224

CSV_DIR="dataset"
IMG_DIR="dataset/test"
CHECKPOINT_PATH="output/${MODEL_NAME}_lr${LEARNING_RATE}_bs${BATCH_SIZE}/${MODEL_NAME}_epoch20.pth"
LOG_DIR="logs"

# ─── Run Evaluation ───────────────────────────────────────────
python3 test.py \
--model "$MODEL_NAME" \
--user "$STUDENT_NAME" \
--batch_size "$BATCH_SIZE" \
--learning_rate "$LEARNING_RATE" \
--csv_root_dir "$CSV_DIR" \
--img_dir "$IMG_DIR" \
--train_datacsv "train.csv" \
--val_datacsv "val.csv" \
--test_datacsv "test.csv" \
--weights "DEFAULT" \
--n_classes "$N_CLASSES" \
--resized_img_weight "$RESIZE" \
--resized_img_height "$RESIZE" \
--patch_size "$PATCH_SIZE" \
--saved_checkpoint_path "$CHECKPOINT_PATH" \
--log_dir "$LOG_DIR"
