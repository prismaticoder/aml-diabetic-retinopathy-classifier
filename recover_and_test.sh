#!/bin/bash

MODELS_DIR="output"
LOGS_DIR="logs"
CSV_ROOT="dataset"
IMG_DIR="dataset/train"
VAL_CSV="dataset/val.csv"
USER="Zohaib Shaikh"
STUDENT_ID="6891120"
PYTHON=python3

echo "📁 Scanning for models to recover logs and test..."

for MODEL_DIR in $MODELS_DIR/*; do
  if [[ -d "$MODEL_DIR" ]]; then
    BASENAME=$(basename "$MODEL_DIR")

    # Parse model details
    MODEL=$(echo $BASENAME | cut -d'_' -f1)
    LR=$(echo $BASENAME | grep -oP 'lr\K[\d\.]+')
    BS=$(echo $BASENAME | grep -oP 'bs\K\d+')
    LOSS=$(echo $BASENAME | grep -oP 'loss\K[^_]+')
    AUG=$(echo $BASENAME | grep -oP 'aug\K[^_]+')
    OPTIM=$(echo $BASENAME | grep -oP '_opt\K[^_]+')

    LOG_NAME="${USER// /_}_ID${STUDENT_ID}_${MODEL}_train_${BS}_${LR}.json"
    LOG_PATH="$LOGS_DIR/$LOG_NAME"

    # Check for checkpoints
    CKPT_COUNT=$(ls "$MODEL_DIR"/*.pth 2>/dev/null | wc -l)

    if [[ $CKPT_COUNT -gt 0 && ! -f "$LOG_PATH" ]]; then
      echo "🔄 Reconstructing log: $LOG_PATH from $CKPT_COUNT checkpoints"
      $PYTHON scripts/rebuild_log_from_checkpoints.py \
        --model "$MODEL" \
        --model_dir "$MODEL_DIR" \
        --user "$USER" \
        --batch_size "$BS" \
        --learning_rate "$LR" \
        --optim "$OPTIM" \
        --loss "$LOSS" \
        --augmentation "$AUG" \
        --weights "DEFAULT" \
        --csv_root_dir "$CSV_ROOT" \
        --img_dir "$IMG_DIR" \
        --val_datacsv "$VAL_CSV" \
        --log_output_path "$LOG_PATH"
    else
      echo "✅ Log exists or no checkpoints: $BASENAME"
    fi

    # Run test if test log is missing
    TEST_LOG="logs/${USER// /_}_ID${STUDENT_ID}_${MODEL}_test_${BS}_${LR}.json"
    if [[ ! -f "$TEST_LOG" ]]; then
      echo "🧪 Running test for $MODEL | BS=$BS | LR=$LR | AUG=$AUG"
      bash test.sh --model "$MODEL" \
        --batch_size "$BS" \
        --learning_rate "$LR" \
        --augment "$AUG"
    else
      echo "✅ Test log already exists for $MODEL (BS=$BS, LR=$LR)"
    fi
  fi
done
