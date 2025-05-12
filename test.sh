#!/bin/bash

# ----------------------- STUDENT INFO -----------------------
STUDENT_ID=6891120 
STUDENT_NAME="Zohaib Shaikh"

# ----------------------- INPUT ARGS -------------------------
MODEL_NAME=$1
BATCH_SIZE=$2
LEARNING_RATE=$3
AUG_PROFILE=$4

# Default fallback if not provided
MODEL_NAME=${MODEL_NAME:-"rsgnet"}
BATCH_SIZE=${BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-"0.0001"}
AUG_PROFILE=${AUG_PROFILE:-"none"}

# ---------------------- VARIANT MAPPING ---------------------
if [[ "$MODEL_NAME" == "rsgnet_removed" ]]; then
  BASE_MODEL="rsgnet"
  MODEL_VARIANT="remove_layer"
elif [[ "$MODEL_NAME" == "rsgnet_added" ]]; then
  BASE_MODEL="rsgnet"
  MODEL_VARIANT="added_layer"
elif [[ "$MODEL_NAME" == "rsgnet_avg_best" ]]; then
  BASE_MODEL="rsgnet"
  MODEL_VARIANT="avgpool"
else
  BASE_MODEL="$MODEL_NAME"
  MODEL_VARIANT="baseline"
fi

# ----------------------- WEIGHTS MAP ------------------------
if [ "$BASE_MODEL" == "efficientnet_v2_s" ]; then
  WEIGHTS="EfficientNet_V2_S_Weights.IMAGENET1K_V1"
elif [ "$BASE_MODEL" == "resnet50" ]; then
  WEIGHTS="ResNet50_Weights.IMAGENET1K_V1"
elif [ "$BASE_MODEL" == "efficientnet" ]; then
  WEIGHTS="EfficientNet_B0_Weights.IMAGENET1K_V1"
else
  WEIGHTS="DEFAULT"
fi

# --------------------- CHECKPOINT PATH ----------------------
CHECKPOINT_DIR="output/${MODEL_NAME}_optadam_lr${LEARNING_RATE}_bs${BATCH_SIZE}_losscrossentropy_aug${AUG_PROFILE}"
CHECKPOINT_PATH=$(ls -t ${CHECKPOINT_DIR}/*.pth 2>/dev/null | head -n 1)

if [ -z "$CHECKPOINT_PATH" ]; then
  echo "❌ No checkpoint found in $CHECKPOINT_DIR"
  exit 1
fi

# ---------------------- CONF MATRIX TITLE -------------------
CONF_MATRIX_TITLE="Confusion Matrix: ${MODEL_NAME} | BS=${BATCH_SIZE} | LR=${LEARNING_RATE} | AUG=${AUG_PROFILE}"

# ---------------------- RUN TEST SCRIPT ---------------------
echo "🧪 Running test for $MODEL_NAME (BS=$BATCH_SIZE, LR=$LEARNING_RATE, AUG=$AUG_PROFILE)"
python test.py \
  --model $BASE_MODEL \
  --user "$STUDENT_NAME" \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --csv_root_dir dataset \
  --img_dir dataset/train \
  --log_dir logs \
  --evaluate_only \
  --saved_checkpoint_path "$CHECKPOINT_PATH" \
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
  --confusion_matrix_title "$CONF_MATRIX_TITLE" \
  --model_variant "$MODEL_VARIANT"
