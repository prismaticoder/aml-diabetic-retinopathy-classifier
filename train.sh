#!/bin/bash

# ====== Student & Model Info ======
STUDENT_ID=6896375
STUDENT_NAME="Lawrence Attoh"
MODEL_NAME="mlp_mixer_v2"
LEARNING_RATE="0.0003"
BATCH_SIZE="32"
WEIGHTS="DEFAULT"
CHECKPOINT_PATH="${MODEL_NAME}_lr${LEARNING_RATE}_bs${BATCH_SIZE}"

# ====== Launch Training ======
python3 train.py \
--model $MODEL_NAME \
--user "$STUDENT_NAME" \
--batch_size $BATCH_SIZE \
--learning_rate $LEARNING_RATE \
--csv_root_dir dataset \
--img_dir dataset/train \
--epochs 20 \
--n_classes 5 \
--optim "adam" \
--lr_scheduler "CosineAnnealingLR" \
--seed 42 \
--brightness 0.2 \
--contrast 0.2 \
--saturation 0.2 \
--hue 0.2 \
--resized_img_weight 224 \
--resized_img_height 224 \
--train_datacsv "dataset/train.csv" \
--val_datacsv "dataset/val.csv" \
--test_datacsv "dataset/test.csv" \
--saved_checkpoint_path $CHECKPOINT_PATH \
--data_augmentation \
--save_model_every_epoch \
--early_stopping \
--log_dir logs \
--weights $WEIGHTS
