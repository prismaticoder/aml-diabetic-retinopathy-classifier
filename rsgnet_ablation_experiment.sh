#!/bin/bash

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

declare -A ARCH_MAP
ARCH_MAP["baseline"]="rsgnet"
ARCH_MAP["removed"]="rsgnet_removed"
ARCH_MAP["added"]="rsgnet_added"
ARCH_MAP["avg"]="rsgnet_avg_best"

declare -A QWK_RESULTS

run_experiment() {
  ARCH=$1
  echo "🚀 Running: $ARCH"
  python3 train.py \
    --model $ARCH \
    --user "$STUDENT_NAME" \
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
    --save_model_every_epoch \
    --early_stopping \
    --data_augmentation
  echo "✅ Done: $ARCH"
  echo "--------------------------------------"
}

# Run baseline and ablation variants
for key in baseline removed added; do
  run_experiment ${ARCH_MAP[$key]}
done

# Parse QWK from logs
for key in baseline removed added; do
  LOG_FILE="logs/${STUDENT_NAME// /_}_ID${STUDENT_ID}_${ARCH_MAP[$key]}_train_${BS}_${LR}.json"
  if [ -f "$LOG_FILE" ]; then
    QWK=$(jq '.epoch_logs[-1].val_qwk' "$LOG_FILE")
    QWK_RESULTS[$key]=$QWK
  else
    QWK_RESULTS[$key]=0
  fi
done

# Select best
BEST_KEY="removed"
if (( $(echo "${QWK_RESULTS[added]} > ${QWK_RESULTS[removed]}" | bc -l) )); then
  BEST_KEY="added"
fi

# Save decision
echo "📊 Selecting best variant based on validation QWK:" > "$LOG_DIR/rsgnet_ablation_best_model.txt"
for key in baseline removed added; do
  echo "• $key: QWK = ${QWK_RESULTS[$key]}" >> "$LOG_DIR/rsgnet_ablation_best_model.txt"
done
echo "✅ Best performing model: ${ARCH_MAP[$BEST_KEY]}" >> "$LOG_DIR/rsgnet_ablation_best_model.txt"
echo "Reason: Highest QWK score among ablation experiments." >> "$LOG_DIR/rsgnet_ablation_best_model.txt"

# Run average pooling using the best
echo "🎯 Running average pooling variant using best model: ${ARCH_MAP[$BEST_KEY]} → ${ARCH_MAP["avg"]}"
run_experiment ${ARCH_MAP["avg"]}
