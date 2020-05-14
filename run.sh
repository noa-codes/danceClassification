#!/bin/bash

if [ "$1" = "test_encoding" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode 1 --model baseline_lstm --train-path=/mnt/disks/disk1/processed/rgb_train_index.csv --val-path=/mnt/disks/disk1/processed/rgb_val_index.csv --batch-size 100
fi