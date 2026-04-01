#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python3 scripts/start_testing.py --experiments_json hierswin_alpha0.4/opts.json \
    --checkpoint_path checkpoint/cervix.pth.tar \
    --test-csv dataset/HierSwin_res.csv \
    --batch-size 32 \
    --visualization
