#!/bin/bash

# This script is used to run the training for the next 3 experiments
# In this experiment, ablation is done on the added layer of the RSGNET model
# The added layer is removed from the model's architecture
# The following ablation variants are trained:
# 1. rsgnet_added - added layer is used
# 2. rsgnet_removed - existing layer is removed
# 3. rsgnet_avg_best - best model from the added and removed variants is used

# Go two levels up from the current script's directory
cd "$(dirname "$0")/../.."

STUDENT_ID=6891120
STUDENT_NAME="Zohaib Shaikh"
MODEL_NAME="rsgnet"
LR="0.0001"
BS=32
OPTIMIZER="adam"
LOSS="crossentropy"
EPOCHS=20
CSV_ROOT="dataset"
IMG_DIR="dataset/train"
TRAIN_CSV="dataset/train.csv"
VAL_CSV="dataset/val.csv"
TEST_CSV="dataset/test.csv"
LOG_DIR="logs"
IMG_WIDTH=224
IMG_HEIGHT=224
AUG="none"
WEIGHTS="DEFAULT"
LR_SCHEDULER="CosineAnnealingLR"

# Map for model_variant
declare -A VARIANT_MAP
VARIANT_MAP["rsgnet_added"]="added_layer"
VARIANT_MAP["rsgnet_avg_best"]="avgpool"

# Target variants (only the ones needed now)
VARIANTS=("rsgnet_added")

declare -A QWK_RESULTS

run_experiment() {
  ARCH=$1
  VARIANT=${VARIANT_MAP[$ARCH]}
  LOG_NAME="${STUDENT_NAME// /_}_ID${STUDENT_ID}_${ARCH}_train_${BS}_${LR}.json"

  if [ -f "logs/$LOG_NAME" ]; then
    echo "⏭️ Skipping $ARCH (already exists: $LOG_NAME)"
    return
  fi

  echo "🚀 Running: $ARCH"
  python train.py \
    --model $ARCH \
    --user "$STUDENT_NAME" \
    --student_id $STUDENT_ID \
    --batch_size $BS \
    --learning_rate $LR \
    --csv_root_dir $CSV_ROOT \
    --img_dir $IMG_DIR \
    --epochs $EPOCHS \
    --n_classes 5 \
    --optim $OPTIMIZER \
    --loss $LOSS \
    --lr_scheduler $LR_SCHEDULER \
    --seed 42 \
    --resized_img_weight $IMG_WIDTH \
    --resized_img_height $IMG_HEIGHT \
    --train_datacsv $TRAIN_CSV \
    --val_datacsv $VAL_CSV \
    --test_datacsv $TEST_CSV \
    --saved_checkpoint_path "${ARCH}_lr${LR}_bs${BS}" \
    --log_dir $LOG_DIR \
    --weights $WEIGHTS \
    --augmentation_profile $AUG \
    --model_variant $VARIANT \
    --save_model_every_epoch \
    --early_stopping \
    --data_augmentation

  echo "✅ Done: $ARCH"
  echo "--------------------------------------"
}

# Run rsgnet_added only (as baseline + removed are already trained)
for model_key in "${VARIANTS[@]}"; do
  run_experiment $model_key
done

# Python QWK extractor
extract_qwk() {
  local json_path=$1
  python -c "
import json
with open('$json_path') as f:
    data = json.load(f)
print(data['epoch_logs'][-1]['val_qwk'])
"
}

# Best selection logic (between added and removed)
BEST_KEY=""
BEST_QWK=-1

for key in "rsgnet_removed" "rsgnet_added"; do
  LOG_FILE="logs/${STUDENT_NAME// /_}_ID${STUDENT_ID}_${key}_train_${BS}_${LR}.json"
  if [ -f "$LOG_FILE" ]; then
    QWK=$(extract_qwk "$LOG_FILE")
    QWK_RESULTS[$key]=$QWK
    if (( $(echo "$QWK > $BEST_QWK" | bc -l) )); then
      BEST_QWK=$QWK
      BEST_KEY=$key
    fi
  else
    QWK_RESULTS[$key]=0
  fi
done

# Save result to file
RESULT_TXT="$LOG_DIR/rsgnet_ablation_best_model.txt"
echo "📊 Validation QWK comparison:" > "$RESULT_TXT"
for key in "rsgnet_removed" "rsgnet_added"; do
  echo "• $key: QWK = ${QWK_RESULTS[$key]}" >> "$RESULT_TXT"
done
echo "✅ Best performing: $BEST_KEY" >> "$RESULT_TXT"
echo "Reason: Highest QWK among ablation variants." >> "$RESULT_TXT"
cat "$RESULT_TXT"

# Run avgpool variant on best
echo "🎯 Running average pooling variant using best model: $BEST_KEY → rsgnet_avg_best"
run_experiment rsgnet_avg_best