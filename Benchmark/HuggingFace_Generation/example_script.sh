#!/usr/bin/env bash



python3 run_hf.py \
    --model_name_or_path google_pegasus-xsum\
    --do_train \
    --seed=88 \
    --save_total_limit=1 \
    --train_file SG/train.json \
    --validation_file SG/val.json \
    --output_dir SG_Pretrained_Pegasus \
    --per_device_train_batch_size=16 \
    --overwrite_output_dir \
    --predict_with_generate

python3 /run_inference.py \
      -t SG \
      -m SG_Pretrained_Pegasus \
      -s G_Pretrained_Pegasus_pred \
      -d SG \
      --model_name pegasus \
      -b 16




























