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
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --mode=tune --epochs=200 --model=baseline_lstm --log=tune_lstm --ntrials=30
elif [ "$1" = "tune_lstm_attention" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --mode=tune --epochs=200 --model=attention_lstm --log=tune_lstm_attention
elif [ "$1" = "tune_tcn" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --mode=tune --epochs=200 --model=tcn --log=tune_tcn --ntrials=50
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
    # hyperparameters from tuning
    CUDA_VISIBLE_DEVICES=0 python3 run.py --model=tcn --encode=0 --batch-size=128 --log=tcn --learning-rate=.02022 --epochs=500 --hidden-size=128 --levels 8 --optim=SGD --dropout 0.05
elif [ "$1" = "train_rnn_attention" ]; then
    # hyperparameters from tuning
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --batch-size=16 --log=lstm_att_tuned --learning-rate=0.0391033 --hidden-size=32 --optimizer=SGD --epochs=200 --model=attention_lstm

######################################
# TESTING
######################################
elif [ "$1" = "test_rnn" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --hidden-size=64  --mode=test --log=test_rnn --checkpoint=/mnt/disks/disk1/log/lstm_tuned_2020_6_4_h2_m34_lr0.004/checkpoints/best_val_loss.pth
elif [ "$1" = "test_tcn" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --hidden-size=128 --mode=test --model=tcn --log=test_tcn --levels=6 --checkpoint=/mnt/disks/disk1/log/tcn_2020_6_5_h23_m7_lr0.002187/checkpoints/best_val_loss.pth
elif [ "$1" = "test_rnn_attention" ]; then
    CUDA_VISIBLE_DEVICES=0 python3 run.py --encode=0 --hidden-size=32  --mode=test --log=test_rnn_att --model=attention_lstm --checkpoint=/mnt/disks/disk1/log/lstm_att_tuned_2020_6_4_h18_m12_lr0.0391033/checkpoints/best_val_loss.pth
fi
