#!/bin/bash

# Student Info
STUDENT_ID=6891120
STUDENT_NAME="Zohaib Shaikh"

# Fixed Parameters
MODEL_NAME="rsgnet"
LR_SCHEDULER="CosineAnnealingLR"
EPOCHS=20
N_CLASSES=5
SEED=42
CSV_ROOT="dataset"
IMG_DIR="dataset/train"
TRAIN_CSV="dataset/train.csv"
VAL_CSV="dataset/val.csv"
TEST_CSV="dataset/test.csv"
LOG_DIR="logs"
IMG_WIDTH=224
IMG_HEIGHT=224
WEIGHTS="DEFAULT"

# Expanded Sweep Variables
BATCH_SIZE=8
LEARNING_RATES=("0.0001" "0.0003" "0.0005" "0.001")
OPTIMIZERS=("adam" "sgd" "adamw")
LOSSES=("crossentropy" "focal" "labelsmoothing")
AUGMENTATIONS=("none" "basic" "advanced")

for LR in "${LEARNING_RATES[@]}"; do
  for OPTIM in "${OPTIMIZERS[@]}"; do
    for LOSS in "${LOSSES[@]}"; do
      for AUG in "${AUGMENTATIONS[@]}"; do

        RUN_NAME="${MODEL_NAME}_opt${OPTIM}_lr${LR}_bs${BATCH_SIZE}_loss${LOSS}_aug${AUG}"

        echo " Running: $RUN_NAME"

        python3 train.py \
          --model $MODEL_NAME \
          --user "$STUDENT_NAME" \
          --batch_size $BATCH_SIZE \
          --learning_rate $LR \
          --csv_root_dir $CSV_ROOT \
          --img_dir $IMG_DIR \
          --epochs $EPOCHS \
          --n_classes $N_CLASSES \
          --optim $OPTIM \
          --loss $LOSS \
          --lr_scheduler $LR_SCHEDULER \
          --seed $SEED \
          --brightness 0.2 \
          --contrast 0.2 \
          --saturation 0.2 \
          --hue 0.2 \
          --resized_img_weight $IMG_WIDTH \
          --resized_img_height $IMG_HEIGHT \
          --train_datacsv $TRAIN_CSV \
          --val_datacsv $VAL_CSV \
          --test_datacsv $TEST_CSV \
          --saved_checkpoint_path $RUN_NAME \
          --save_model_every_epoch \
          --early_stopping \
          --data_augmentation \
          --augmentation_profile $AUG \
          --log_dir $LOG_DIR \
          --weights $WEIGHTS

        echo "✅ Finished: $RUN_NAME"
        echo "----------------------------------------"

      done
    done
  done
done
