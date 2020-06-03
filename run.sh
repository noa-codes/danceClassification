#!/bin/bash

if [ "$1" = "pose_encode" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=3 --batch-size=100 --log=pose --learning-rate=1e-3
elif [ "$1" = "train_rnn" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --batch-size=20 --log=lstm --learning-rate=1e-3 --epochs=200
elif [ "$1" = "test_rnn" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --batch-size=100 --mode=test --checkpoint=/mnt/disks/disk1/log/lstm_2020_5_21_h17_m6_lr0.001/checkpoints/best_val_loss.pth
elif [ "$1" = "train_tcn" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --batch-size=20 --log=tcn --learning-rate=1e-3 --epochs=500 --hidden-size=25 --levels 3 --optim=Adam --dropout 0.25
elif [ "$1" = "debug_tuning" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --mode=tune --epochs=200 --model=baseline_lstm --log=tune_lstm
fi
