#!/bin/bash

# ─── Meta Info ────────────────────────────────────────────────
STUDENT_NAME="Lawrence Attoh"
MODEL_NAME="mlp_mixer_v2_addblock"
LEARNING_RATE="0.0005"
BATCH_SIZE="128"
EPOCHS="20"
PATCH_SIZE="16"
CSV_DIR="dataset"
IMG_DIR="dataset/train"
LOG_DIR="logs"

# ─── Run Training ─────────────────────────────────────────────
python3 train.py \
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
--n_classes 5 \
--resized_img_weight 224 \
--resized_img_height 224 \
--log_dir "$LOG_DIR" \
--epochs "$EPOCHS" \
--seed 42 \
--optim "adamw" \
--lr_scheduler "cosine" \
--patch_size "$PATCH_SIZE"
