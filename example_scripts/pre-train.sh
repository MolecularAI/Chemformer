#!/bin/bash

python setup.py develop
python -m molbart.train \
  --dataset_type zinc \
  --data_path ../data/zinc \
  --model_type bart \
  -lr 1.0 \
  --schedule transformer \
  --n_epochs 6 \
  --batch_size 128 \
  --d_model 1024 \
  --n_layers 8 \
  --n_heads 16 \
  --d_feedforward 4096 \
  --n_gpus 4 \
  --task mask_aug