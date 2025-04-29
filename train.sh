#!/bin/bash

# Set student info
export STUDENT_ID=6896375 
export STUDENT_NAME="Lawrence Attoh"

# MLP-Mixer model training setup
export MODEL_NAME="mlp_mixer"
export LEARNING_RATE="0.0001"
export BATCH_SIZE="64"
export CHECKPOINT_PATH="mlp_mixer_lr${LEARNING_RATE}_bs${BATCH_SIZE}"
export WEIGHTS="DEFAULT"

# Activate your virtual environment (optional: only if needed)
# source venv/bin/activate

# Run training
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
--brightness 0.5 \
--contrast 0.5 \
--saturation 0.5 \
--hue 0.5 \
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
