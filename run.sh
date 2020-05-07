#!/bin/bash

# if [ "$1" = "train_m4_baseline" ]; then
#     CUDA_VISIBLE_DEVICES=0 python3 run.py --model baseline_lstm --train-path=./data/mbounded-dyck-k/m4/train.formal.txt --valid-path=./data/mbounded-dyck-k/m4/dev.formal.txt --hidden-size 12 --embedding-size 30 --log baseline_m4_batch10 --log-every 10 --epochs 1000 --lr 1e-4 --batch-size 10 --dropout 0 --num-layers 1 --is-stream 0