#!/bin/bash
cd ./HiCervix/HierSwin/ 
which python
#export CUDA_VISIBLE_DEVICES=$1
echo $1

echo "python train.py --opt ./configs/$2.yaml"
CUDA_VISIBLE_DEVICES=0 ./envs/hyper/bin/python scripts/start_training.py --arch swinT --loss hierarchical-cross-entropy  --alpha 0.4 --output ablation/batch_256 --train-csv dataset/train.csv --val-csv dataset/val.csv --batch-size 256 --epochs 100 #--target_size 224



