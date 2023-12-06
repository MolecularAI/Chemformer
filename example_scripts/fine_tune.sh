#!/bin/bash

python -m molbart.fine_tune \
  --dataset_type uspto_50 \
  --data_path data/uspto_50.pickle \
  --model_path models/bart/span_aug.ckpt \
  --task backward_prediction \
  --n_epochs 100 \
  -lr 0.001 \
  --schedule cycle \
  --batch_size 128 \
  --acc_batches 4 \
  --augmentation_strategy all \
  -aug_prob 0.5