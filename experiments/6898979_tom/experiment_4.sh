#!/bin/bash

# This script is used to run the training for the fourth experiment
# In this experiment, ablation is done on the grid attention of the maxvit model
# The entire grid attention is removed from the model's architecture leaving just MBConv blocks and block attention

# Go two levels up from the current script's directory
cd "$(dirname "$0")/../.."

# Set student information
STUDENT_ID=6898979 
STUDENT_NAME="Jesutomiwa Salam"

# Set model-specific variables
MODEL_NAME="maxvit"
MODEL_VARIANT="remove_grid_attn"
LEARNING_RATE="1e-4"
BATCH_SIZE="32"

# Define checkpoint model path dynamically based on model name, learning rate, and batch size
CHECKPOINT_PATH="${MODEL_NAME}_lr${LEARNING_RATE}_bs${BATCH_SIZE}"

# Python command to run the training
python train.py \
--model $MODEL_NAME \
--user "$STUDENT_NAME" \
--student_id $STUDENT_ID \
--batch_size $BATCH_SIZE \
--learning_rate $LEARNING_RATE \
--csv_root_dir dataset \
--img_dir dataset/train \
--epochs 30 \
--n_classes 5 \
--optim "adamw" \
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
--log_dir logs \
--loss "crossentropy" \
--model_variant $MODEL_VARIANT