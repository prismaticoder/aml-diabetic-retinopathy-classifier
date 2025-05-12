#!/bin/bash

# Usage: bash test.sh <model_name> <batch_size> <learning_rate> <augmentation_profile>
MODEL=$1
BS=$2
LR=$3
AUG=$4

echo "🧪 Running test for $MODEL (BS=$BS, LR=$LR, AUG=$AUG)"
python3 test.py \
  --model "$MODEL" \
  --user "Zohaib Shaikh" \
  --batch_size "$BS" \
  --learning_rate "$LR" \
  --csv_root_dir dataset \
  --img_dir dataset/train \
  --n_classes 5 \
  --optim "adam" \
  --lr_scheduler "CosineAnnealingLR" \
  --seed 42 \
  --resized_img_weight 224 \
  --resized_img_height 224 \
  --train_datacsv "dataset/train.csv" \
  --val_datacsv "dataset/val.csv" \
  --test_datacsv "dataset/test.csv" \
  --saved_checkpoint_path "output/${MODEL}_optadam_lr${LR}_bs${BS}_losscrossentropy_aug${AUG}" \
  --log_dir logs \
  --weights "DEFAULT" \
  --augmentation_profile "$AUG" \
  --confusion_matrix_title "Confusion Matrix for ${MODEL}" \
  --evaluate_only
