#!/bin/bash

# This script is used to run the training for the first experiment
# It is used to train the maxvit model with the default settings used for the baseline model
# All blocks and stages of the maxvit model are used in this stage

# Set student information
STUDENT_ID=6898979 
STUDENT_NAME="Jesutomiwa Salam"

# Set model-specific variables
MODEL_NAME="maxvit"
LEARNING_RATE="1e-4"
BATCH_SIZE="32"

# Define checkpoint model path dynamically based on model name, learning rate, and batch size
CHECKPOINT_PATH="${MODEL_NAME}_lr${LEARNING_RATE}_bs${BATCH_SIZE}"

# Python command to run the training
python ../../train.py \
--model $MODEL_NAME \
--user "$STUDENT_NAME" \
--student_id $STUDENT_ID \
--batch_size $BATCH_SIZE \
--learning_rate $LEARNING_RATE \
--csv_root_dir labels \
--img_dir dataset/train \
--epochs 20 \
--n_classes 5 \
--optim "adam" \
--lr_scheduler "CosineAnnealingLR" \
--seed 42 \
--resized_img_weight 224 \
--resized_img_height 224 \
--train_datacsv "labels/train.csv" \
--val_datacsv "labels/val.csv" \
--test_datacsv "labels/test.csv" \
--saved_checkpoint_path $CHECKPOINT_PATH \
--data_augmentation \
--save_model_every_epoch \
--early_stopping \
--log_dir logs
