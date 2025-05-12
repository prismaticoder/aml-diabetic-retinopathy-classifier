#!/bin/bash

# Set student information
STUDENT_ID=6904186 
STUDENT_NAME="Meenakshy Prem"

# Set model-specific variables
MODEL_NAME="swin_custom"
LEARNING_RATE="0.0005"
BATCH_SIZE="16"

# Dynamically assign the weights based on model name
if [ "$MODEL_NAME" == "efficientnet_v2_s" ]; then
  WEIGHTS="EfficientNet_V2_S_Weights.IMAGENET1K_V1"
elif [ "$MODEL_NAME" == "resnet50" ]; then
  WEIGHTS="ResNet50_Weights.IMAGENET1K_V1"
else
  WEIGHTS="DEFAULT"
fi

# Define checkpoint model path dynamically based on model name, learning rate, and batch size
CHECKPOINT_DIR="output/${MODEL_NAME}_lr${LEARNING_RATE}_bs${BATCH_SIZE}"
CHECKPOINT_PATH=$(ls -t ${CHECKPOINT_DIR}/*.pth | head -n 1)  # Get the latest .pth file

# Build Confusion Matrix Title based on model name
CONF_MATRIX_TITLE="Confusion Matrix for $(echo ${MODEL_NAME} | tr '_' ' ' | sed 's/\b\(.\)/\u\1/g')"

# Run the test script
python3 test.py \
--model $MODEL_NAME \
--user "$STUDENT_NAME" \
--batch_size $BATCH_SIZE \
--learning_rate $LEARNING_RATE \
--csv_root_dir dataset \
--img_dir dataset/train \
--log_dir logs \
--evaluate_only \
--saved_checkpoint_path $CHECKPOINT_PATH \
--n_classes 5 \
--resized_img_weight 224 \
--resized_img_height 224 \
--seed 42 \
--optim "adam" \
--lr_scheduler "CosineAnnealingLR" \
--train_datacsv "dataset/train.csv" \
--val_datacsv "dataset/val.csv" \
--test_datacsv "dataset/test.csv" \
--weights $WEIGHTS \
--confusion_matrix_title "$CONF_MATRIX_TITLE"
