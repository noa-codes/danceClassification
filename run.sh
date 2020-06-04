#!/bin/bash

######################################
# ENCODING
######################################
if [ "$1" = "pose_encode" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=3 --batch-size=100 --log=pose --learning-rate=1e-3
elif [ "$1" = "train_rnn" ]; then
    # hyperparameters from tuning
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --batch-size=32 --log=lstm_tuned --learning-rate=4e-3 --epochs=500 --hidden-size=64 --mode=train --optimizer=SGD
elif [ "$1" = "test_rnn" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --batch-size=100 --mode=test --checkpoint=/mnt/disks/disk1/log/lstm_2020_5_21_h17_m6_lr0.001/checkpoints/best_val_loss.pth
elif [ "$1" = "train_tcn" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --model=tcn --encode=0 --batch-size=20 --log=tcn --learning-rate=1e-3 --epochs=500 --hidden-size=25 --levels 8 --optim=SGD --dropout 0.05
elif [ "$1" = "train_rnn_attention" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --batch-size=20 --log=lstm_att --learning-rate=1e-3 --epochs=200 --model=attention_lstm
elif [ "$1" = "tune_lstm" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --mode=tune --epochs=200 --model=baseline_lstm --log=tune_lstm --ntrials=30
elif [ "$1" = "tune_lstm_attention" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --mode=tune --epochs=200 --model=attention_lstm --log=tune_lstm_attention
elif [ "$1" = "tune_tcn" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --mode=tune --epochs=200 --model=tcn --log=tune_tcn
elif [ "$1" = "tune_frame" ]; then
    # hyperparameters from tuning of LSTM
    # before running this, set all hyperparameters to tunable=False except for frame_freq, unless you want to tune many
    # hyperparameters at once
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --mode=tune --epochs=500 --model=baseline_lstm --log=frame_tuned --batch-size=32 --learning-rate=4e-3 --hidden-size=64 --optimizer=SGD

######################################
# TRAINING
######################################
elif [ "$1" = "train_rnn" ]; then
    # hyperparameters from tuning
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --batch-size=32 --log=lstm_tuned --learning-rate=4e-3 --epochs=500 --hidden-size=64 --mode=train --optimizer=SGD
elif [ "$1" = "train_tcn" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --model=tcn --encode=0 --batch-size=16 --log=tcn --learning-rate=.1 --epochs=500 --hidden-size=32 --levels 8 --optimizer=SGD --dropout 0.05
    # CUDA_VISIBLE_DEVICES=0 python3 run.py --model=tcn --encode=0 --batch-size=20 --log=tcn --learning-rate=1e-3 --epochs=500 --hidden-size=25 --levels 8 --optimizer=SGD --dropout 0.05
elif [ "$1" = "train_rnn_attention" ]; then
    # hyperparameters from tuning
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --batch-size=16 --log=lstm_att_tuned --learning-rate=0.0391033 --hidden-size=32 --optimizer=SGD --epochs=200 --model=attention_lstm

######################################
# TESTING
######################################
elif [ "$1" = "test_rnn" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --batch-size=100 --mode=test --checkpoint=/mnt/disks/disk1/log/lstm_2020_5_21_h17_m6_lr0.001/checkpoints/best_val_loss.pth
fi
