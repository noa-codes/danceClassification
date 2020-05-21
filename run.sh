#!/bin/bash

if [ "$1" = "pose_encode" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=3 --batch-size=100 --log=pose --learning-rate=3e-5
elif [ "$1" = "train_rnn" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --batch-size=10 --log=lstm --learning-rate=1e-3 --epochs=100
elif [ "$1" = "test_rnn" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --batch-size=10 --mode=test --checkpoint=/mnt/disks/disk1/log/lstm_2020_5_21_h1_m23_lr0.001/checkpoints/best_val_loss.pth
fi
