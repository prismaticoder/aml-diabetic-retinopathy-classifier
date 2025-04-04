#!/bin/bash
python test.py \
--model resnet50 \
--user zohaib \
--batch_size 32 \
--learning_rate 0.0001 \
--csv_file dataset/val.csv \
--img_dir dataset/train
