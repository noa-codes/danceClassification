#!/bin/bash

if [ "$1" = "debug" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --model baseline_lstm --image-train-path=/mnt/disks/disk1/processed/rgb/rgb_train_index.csv --image-val-path=/mnt/disks/disk1/processed/rgb/rgb_val_index.csv --encode-path=/mnt/disks/disk1/processed/rgb --pose-train-path=/mnt/disks/disk1/processed/densepose/densepose_train_index.csv --pose-val-path=/mnt/disks/disk1/processed/densepose/densepose_val_index.csv --batch-size 100
fi
