#!/bin/bash

# Go two levels up from the current script's directory
cd "$(dirname "$0")/../.."

# Set student information
STUDENT_ID=6904186
STUDENT_NAME="Meenakshy Prem"

# Set model-specific variables
MODEL_NAME="swin_custom_addextrablocks"
LEARNING_RATE="0.0001"
BATCH_SIZE="32"

# Define checkpoint model path dynamically based on model name, learning rate, and batch size
CHECKPOINT_PATH="${MODEL_NAME}_lr${LEARNING_RATE}_bs${BATCH_SIZE}"

# Python command to run the training
python train.py \
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