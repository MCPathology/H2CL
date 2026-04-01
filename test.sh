#!/bin/bash

if [ $1 -lt 100 ]; then
    checkpoint_name="checkpoint.epoch00$1.pth.tar"
else
    checkpoint_name="checkpoint.epoch0$1.pth.tar"
fi

CUDA_VISIBLE_DEVICES=1 python3 scripts/start_testing.py --experiments_json hierswin_alpha0.4/opts.json \
    --checkpoint_path checkpoint/cervix.pth.tar \
    --test-csv dataset/HierSwin_res.csv \
    --batch-size 32
#   --checkpoint_path ablation/all_swin/model_snapshots/$checkpoint_name \
python3 scripts/metrics_eval.py --res-csv hierswin_alpha0.4/HierSwin_res_res.csv
