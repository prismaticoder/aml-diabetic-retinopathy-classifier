#!/bin/bash

# This script is used to run the training for the eighth experiment
# It involves keeping the learning rate the same and adjusting the batch size to 64 to evaluate the scalability of the baseline RSGNet

STUDENT_ID=6891120
STUDENT_NAME="Zohaib Shaikh"
MODEL_NAME="rsgnet"
LEARNING_RATE="0.0001"
BATCH_SIZE="64"
AUGMENTATION="none"
OPTIMIZER="adam"
LOSS_FUNCTION="crossentropy"
CHECKPOINT_PATH="${MODEL_NAME}_lr${LEARNING_RATE}_bs${BATCH_SIZE}"

python ../../train.py \
  --model $MODEL_NAME \
  --user "$STUDENT_NAME" \
  --student_id $STUDENT_ID \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --csv_root_dir dataset \
  --img_dir dataset/train \
  --epochs 20 \
  --n_classes 5 \
  --optim $OPTIMIZER \
  --loss $LOSS_FUNCTION \
  --lr_scheduler "CosineAnnealingLR" \
  --seed 42 \
  --resized_img_weight 224 \
  --resized_img_height 224 \
  --train_datacsv "dataset/train.csv" \
  --val_datacsv "dataset/val.csv" \
  --test_datacsv "dataset/test.csv" \
  --saved_checkpoint_path $CHECKPOINT_PATH \
  --data_augmentation \
  --augmentation_profile $AUGMENTATION \
  --save_model_every_epoch \
  --early_stopping \
  --log_dir logs \
  --weights DEFAULT \
  --model_variant "baseline"