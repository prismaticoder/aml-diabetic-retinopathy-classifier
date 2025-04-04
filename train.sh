#!/bin/bash
python train.py \
--model resnet50 \
--user zohaib \
--batch_size 32 \
--learning_rate 0.0001 \
--csv_root_dir dataset \
--img_dir dataset/train
