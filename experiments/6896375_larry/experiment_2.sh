#!/bin/bash

# Go two levels up from the current script's directory
cd "$(dirname "$0")/../.."

# ─── Meta ─────────────────────────────────────────────────────
STUDENT_ID=6896375
STUDENT_NAME="Lawrence Attoh"
MODEL_NAME="mlp_mixer_v2_addlayer"
LEARNING_RATE="0.0001"
BATCH_SIZE="32"
EPOCHS="20"
WEIGHTS="DEFAULT"
LOG_DIR="logs"
CSV_DIR="dataset"
IMG_DIR="dataset/train"

# ─── Augmentation Settings ───────────────────────────────────
BRIGHTNESS="0.3"
CONTRAST="0.3"
SATURATION="0.4"
HUE="0.1"

# ─── Run Training ─────────────────────────────────────────────
python train.py \
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
--early_stopping_patience 5
--optim adam \
--lr_scheduler CosineAnnealingLR \
--data_augmentation \
--brightness "$BRIGHTNESS" \
--contrast "$CONTRAST" \
--saturation "$SATURATION" \
--hue "$HUE"