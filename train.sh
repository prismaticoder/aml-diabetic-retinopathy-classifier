#!/bin/bash

# ─── Meta ─────────────────────────────────────────────────────
STUDENT_ID=6896375
STUDENT_NAME="Lawrence Attoh"
MODEL_NAME="mlp_mixer"
LEARNING_RATE="0.0001"
BATCH_SIZE="32"
EPOCHS="20"
WEIGHTS="DEFAULT"
LOG_DIR="logs"
CSV_DIR="dataset"
IMAGE_DIR="dataset/train"
TRAIN_CSV="train.csv"
VAL_CSV="val.csv"
TEST_CSV="test.csv"

# ─── Augmentation Settings ───────────────────────────────────
BRIGHTNESS="0.2"
CONTRAST="0.2"
SATURATION="0.2"
HUE="0.2"

# ─── Run Training ─────────────────────────────────────────────
python train.py \
--model "$MODEL_NAME" \
--user "$STUDENT_NAME" \
--student_id "$STUDENT_ID" \
--batch_size "$BATCH_SIZE" \
--learning_rate "$LEARNING_RATE" \
--epochs "$EPOCHS" \
--csv_root_dir "$CSV_DIR" \
--img_dir "$IMAGE_DIR" \
--train_datacsv "$TRAIN_CSV" \
--val_datacsv "$VAL_CSV" \
--test_datacsv "$TEST_CSV" \
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
