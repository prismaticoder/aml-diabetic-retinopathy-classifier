#!/bin/bash

# This script is used to run the training for the seventh experiment
# In this experiment, the loss function is adjusted to mse while the entire architecture is kept the same. MSE is used
# in such a way that the task can be viewed as an ordinal regression task.

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
--epochs 30 \
--n_classes 5 \
--optim "adamw" \
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
--log_dir logs \
--loss "mse" \
--model_variant "baseline"