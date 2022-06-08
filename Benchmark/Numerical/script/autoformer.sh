#!/bin/bash



python3 cmd_run.py \
  --config_file cfg/SG.cfg \
  --save_path  results/SG_Autoformer \
  --multi_run 5 \
  --seed 42 \
  --embed fixed \
  --model Autoformer \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --train_epochs 20


python3 cmd_run.py \
  --config_file cfg/CT.cfg \
  --save_path  results/CT_Autoformer\
  --multi_run 5 \
  --seed 42 \
  --embed fixed \
  --model Autoformer \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --train_epochs 20

python3 cmd_run.py \
  --config_file cfg/ECL.cfg \
  --save_path  results/ECL_Autoformer \
  --multi_run 5 \
  --seed 42 \
  --embed fixed \
  --model Autoformer \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --train_epochs 20
