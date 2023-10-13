#!/bin/bash

# Activate Python environment (optional)
source /Users/larsmoan/Documents/UQ/COMP3710/PatternAnalysis-2023/.venv/bin/activate

python3 yolov7/train.py  --batch-size 2 --data data/ISIC_2017/isic.yaml --img 32 32 --cfg yolov7_isic_cfg.yaml --weights '' --name yolov7 --hyp hyp.scratch.p5.yaml