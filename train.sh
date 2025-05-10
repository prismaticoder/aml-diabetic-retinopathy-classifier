#!/bin/bash

# Set student information
STUDENT_ID=6898979 
STUDENT_NAME="Jesutomiwa Salam"

# Set model-specific variables
MODEL_NAME="maxvit"
LEARNING_RATE="0.0001"
BATCH_SIZE="32"

# Dynamically assign the weights based on model name
if [ "$MODEL_NAME" == "efficientnet_v2_s" ]; then
  WEIGHTS="EfficientNet_V2_S_Weights.IMAGENET1K_V1"
elif [ "$MODEL_NAME" == "resnet50" ]; then
  WEIGHTS="ResNet50_Weights.IMAGENET1K_V1"
else
  WEIGHTS="DEFAULT"
fi

# Define checkpoint model path dynamically based on model name, learning rate, and batch size
CHECKPOINT_PATH="${MODEL_NAME}_lr${LEARNING_RATE}_bs${BATCH_SIZE}"

# Python command to run the training
python train.py \
--model $MODEL_NAME \
--user "$STUDENT_NAME" \
--batch_size $BATCH_SIZE \
--learning_rate $LEARNING_RATE \
--csv_root_dir labels \
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
--train_datacsv "labels/train.csv" \
--val_datacsv "labels/val.csv" \
--test_datacsv "labels/test.csv" \
--saved_checkpoint_path $CHECKPOINT_PATH \
--data_augmentation \
--save_model_every_epoch \
--early_stopping \
--log_dir logs \
--weights $WEIGHTS
