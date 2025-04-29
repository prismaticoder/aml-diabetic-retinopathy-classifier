#!/bin/bash

STUDENT_ID=6896375
STUDENT_NAME="Lawrence Attoh"
MODEL_NAME="mlp_mixer"
LEARNING_RATE="0.003"
BATCH_SIZE="16"
WEIGHTS="DEFAULT"

CHECKPOINT_DIR="output/${MODEL_NAME}_lr${LEARNING_RATE}_bs${BATCH_SIZE}"
CHECKPOINT_PATH=$(ls -t ${CHECKPOINT_DIR}/*.pth | head -n 1)

CONF_MATRIX_TITLE="Confusion Matrix for $(echo ${MODEL_NAME} | tr '_' ' ' | sed 's/\b\(.\)/\u\1/g')"

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
