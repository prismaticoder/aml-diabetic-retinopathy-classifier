#!/bin/bash

# ─── Meta Info ──────────────────────────────────────────────────────────────
STUDENT_ID=6896375
STUDENT_NAME="Lawrence Attoh"
MODEL_NAME="mlp_mixer_v2_batchnorm"
LEARNING_RATE="0.0003"
BATCH_SIZE="64"
EPOCHS="20"
WEIGHTS="DEFAULT"
LOG_DIR="logs"
CSV_DIR="dataset"
IMG_DIR="dataset/train"

# ─── Augmentation Parameters ────────────────────────────────────────────────
BRIGHTNESS="0.2"
CONTRAST="0.2"
SATURATION="0.2"
HUE="0.2"

# ─── Training ───────────────────────────────────────────────────────────────
python ../../train.py \
--model "$MODEL_NAME" \
--user "$STUDENT_NAME" \
--student_id "$STUDENT_ID" \
--batch_size "$BATCH_SIZE" \
--learning_rate "$LEARNING_RATE" \
--epochs "$EPOCHS" \
--csv_root_dir "$CSV_DIR" \
--img_dir "$IMG_DIR" \
--train_datacsv "train.csv" \
--val_datacsv "val.csv" \
--test_datacsv "test.csv" \
--log_dir "$LOG_DIR" \
--weights "$WEIGHTS" \
--n_classes 5 \
--resized_img_weight 224 \
--resized_img_height 224 \
--seed 42 \
--save_every 2 \
--early_stopping \
--optim adam \
--lr_scheduler CosineAnnealingLR \
--data_augmentation \
--brightness "$BRIGHTNESS" \
--contrast "$CONTRAST" \
--saturation "$SATURATION" \
--hue "$HUE"