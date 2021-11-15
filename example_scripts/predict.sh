#!/bin/bash

python -m molbart.predict \
  --reactants_path uspto_50_test.txt \
  --products_path uspto_50_p0_5_out.pickle \
  --model_path saved_models/uspto_50/span_aug/100_epochs/last.ckpt \
  --batch_size 64 \
  --num_beams 10

