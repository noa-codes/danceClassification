#!/bin/bash

######################################
# ENCODING
######################################
if [ "$1" = "pose_encode" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=3 --batch-size=100 --log=pose --learning-rate=1e-3

######################################
# TUNING
######################################
elif [ "$1" = "tune_lstm" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --mode=tune --epochs=500 --model=baseline_lstm --log=tune_lstm --ntrials=100 --batch-size=64 --patience=10 --optimizer=SGD
elif [ "$1" = "tune_lstm_attention" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --mode=tune --epochs=500 --model=attention_lstm --log=tune_lstm_attention --ntrials=50 --patience=60 --optimizer=SGD
elif [ "$1" = "tune_tcn" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --mode=tune --epochs=200 --model=tcn --log=tune_tcn --ntrials=50
# NOTE: before running tuning on frames, set all hyperparameters to tunable=False except for frame_freq,
# unless you want to tune many hyperparameters at once
elif [ "$1" = "tune_frame" ]; then
    # hyperparameters from tuning of LSTM
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --mode=tune --epochs=500 --model=baseline_lstm --log=frame_tuned --batch-size=32 --learning-rate=4e-3 --hidden-size=64 --optimizer=SGD
elif [ "$1" = "tune_tcn_frames" ]; then
    # hyperparameters from tuning
    CUDA_VISIBLE_DEVICES=0 python3 run.py --model=tcn --encode=0 --mode=tune --batch-size=64 --log=tune_tcn_frames --learning-rate=.02022 --epochs=200 --hidden-size=128 --levels=6 --optim=SGD --dropout=0.02

######################################
# TRAINING
######################################
elif [ "$1" = "train_lstm" ]; then
    # hyperparameters from tuning
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --batch-size=64 --log=lstm_tuned --learning-rate=0.016731 --epochs=500 --hidden-size=128 --mode=train --optimizer=SGD --dropout=0.70 --frame-freq=60 --weight-decay=0.028410 --patience=20
elif [ "$1" = "train_tcn" ]; then
    # hyperparameters from tuning
    CUDA_VISIBLE_DEVICES=0 python3 run.py --model=tcn --encode=0 --batch-size=128 --log=tcn --learning-rate=.02022 --epochs=500 --hidden-size=128 --levels 6 --optim=SGD --dropout 0.05
elif [ "$1" = "train_lstm_attention" ]; then
    # hyperparameters from tuning
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --batch-size=16 --dropout=0.05 --frame-freq=30 --log=lstm_att_tuned --learning-rate=0.059125 --hidden-size=32 --optimizer=SGD --epochs=500 --model=attention_lstm
elif [ "$1" = "train_cnn" ]; then
    # hyperparameters from tuning
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --batch-size=32 --log=cnn --learning-rate=0.01 --hidden-size=256 --optimizer=SGD --epochs=200 --model=baseline_cnn

######################################
# TESTING
######################################
<<<<<<< HEAD
elif [ "$1" = "test_lstm" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --hidden-size=128 --mode=test --log=test_rnn --checkpoint=/mnt/disks/disk1/log/lstm_tuned_2020_6_7_h17_m13_lr0.016731/checkpoints/best_val_loss.pth
elif [ "$1" = "test_tcn" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --hidden-size=128 --mode=test --model=tcn --log=test_tcn --levels=6 --checkpoint=/mnt/disks/disk1/log/tcn_2020_6_5_h23_m7_lr0.002187/checkpoints/best_val_loss.pth
elif [ "$1" = "test_lstm_attention" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --hidden-size=32  --mode=test --log=test_rnn_att --model=attention_lstm --checkpoint=/mnt/disks/disk1/log/lstm_att_tuned_2020_6_4_h18_m12_lr0.0391033/checkpoints/best_val_loss.pth
fi
