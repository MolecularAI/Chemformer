#!/bin/bash

python setup.py develop
python -m molbart.train \
  --dataset zinc \
  --data_path ../data/zinc \
  --model_type bart \
  --lr 1.0 \
  --schedule transformer \
  --epochs 6 \
  --batch_size 128 \
  --d_model 1024 \
  --num_layers 8 \
  --num_heads 16 \
  --d_feedforward 4096 \
  --gpus 4 \
  --task mask_aug

