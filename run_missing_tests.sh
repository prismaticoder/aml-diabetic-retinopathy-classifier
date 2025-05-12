#!/bin/bash

STUDENT_ID=6891120
STUDENT_NAME="Zohaib Shaikh"

CSV_ROOT="dataset"
IMG_DIR="dataset/train"
TRAIN_CSV="${CSV_ROOT}/train.csv"
VAL_CSV="${CSV_ROOT}/val.csv"
TEST_CSV="${CSV_ROOT}/test.csv"
LOG_DIR="logs"

for train_log in logs/*_train_*.json; do
  base=$(basename "$train_log")
  NAME_PART=$(echo "$base" | sed -E 's/Zohaib_Shaikh_ID[0-9]+_//;s/_train_.*//')
  MODEL_NAME=$NAME_PART
  BS=$(echo "$base" | sed -E 's/.*_train_([0-9]+)_.*/\1/')
  LR=$(echo "$base" | sed -E 's/.*_([0-9]+\.[0-9]+)\.json/\1/')
  TEST_LOG="logs/Zohaib_Shaikh_ID${STUDENT_ID}_${MODEL_NAME}_test_${BS}_${LR}.json"

  if [ -f "$TEST_LOG" ]; then
    echo "✅ Test log already exists for $MODEL_NAME (BS=$BS, LR=$LR)"
    continue
  fi

  # Guess best matching model_dir
  CHECKPOINT_DIR=$(find output -maxdepth 1 -type d -name "${MODEL_NAME}_opt*_lr${LR}_bs${BS}_loss*aug*" | head -n 1)

  if [ -z "$CHECKPOINT_DIR" ]; then
    echo "⚠️ No checkpoint found for $MODEL_NAME (BS=$BS, LR=$LR), skipping..."
    continue
  fi

  CHECKPOINT=$(ls -t ${CHECKPOINT_DIR}/*.pth 2>/dev/null | head -n 1)
  if [ -z "$CHECKPOINT" ]; then
    echo "⚠️ No checkpoint file in $CHECKPOINT_DIR"
    continue
  fi

  # Determine augmentation
  AUG="none"
  if [[ "$CHECKPOINT_DIR" == *"augadvanced"* ]]; then
    AUG="advanced"
  elif [[ "$CHECKPOINT_DIR" == *"augbasic"* ]]; then
    AUG="basic"
  fi

  # Weights
  if [[ "$MODEL_NAME" == "efficientnet_v2_s"* ]]; then
    WEIGHTS="EfficientNet_V2_S_Weights.IMAGENET1K_V1"
  elif [[ "$MODEL_NAME" == "resnet50"* ]]; then
    WEIGHTS="ResNet50_Weights.IMAGENET1K_V1"
  else
    WEIGHTS="DEFAULT"
  fi

  echo "🧪 Running test for $MODEL_NAME (BS=$BS, LR=$LR, AUG=$AUG)"
  python3 test.py \
    --model "$MODEL_NAME" \
    --user "$STUDENT_NAME" \
    --batch_size "$BS" \
    --learning_rate "$LR" \
    --csv_root_dir "$CSV_ROOT" \
    --img_dir "$IMG_DIR" \
    --n_classes 5 \
    --optim "adam" \
    --lr_scheduler "CosineAnnealingLR" \
    --seed 42 \
    --resized_img_weight 224 \
    --resized_img_height 224 \
    --train_datacsv "$TRAIN_CSV" \
    --val_datacsv "$VAL_CSV" \
    --test_datacsv "$TEST_CSV" \
    --saved_checkpoint_path "$CHECKPOINT" \
    --log_dir "$LOG_DIR" \
    --weights "$WEIGHTS" \
    --evaluate_only \
    --confusion_matrix_title "Confusion Matrix: $MODEL_NAME"
done
