#!/bin/bash

# ─── Meta Settings ────────────────────────────────────────────────
MODEL_NAME="mlp_mixer_v2_batchnorm"
IMAGE_PATH="dataset/test/10000_left.jpeg"
CHECKPOINT_DIR="output/${MODEL_NAME}_lr0.0003_bs64"
IMAGE_SIZE=224
TARGET_CLASS=1
OUTPUT_DIR="gradcam_outputs"

# ─── Find Latest Checkpoint ───────────────────────────────────────
CHECKPOINT_PATH=$(ls -t ${CHECKPOINT_DIR}/*.pth | head -n 1)
EPOCH_NUM=$(basename "$CHECKPOINT_PATH" | grep -oP 'epoch\K[0-9]+')
OUT_IMAGE="${OUTPUT_DIR}/${MODEL_NAME}_epoch${EPOCH_NUM}_gradcam.png"

# ─── Ensure Output Directory Exists ───────────────────────────────
mkdir -p "$OUTPUT_DIR"

# ─── Run Grad-CAM ─────────────────────────────────────────────────
python3 gradcam_visualizer.py \
  --model "$MODEL_NAME" \
  --weights "DEFAULT" \
  --n_classes 5 \
  --image_path "$IMAGE_PATH" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --output_dir "$OUT_IMAGE" \
  --image_size "$IMAGE_SIZE" \
  --target_class "$TARGET_CLASS"
