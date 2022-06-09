prepare_hf.py: convert PISA dataset train/val files to json format for training HuggingFace models

metrics.py: calculating RMSE, MAE, Missing_Rate of the output predictions.

run_hf.py: the script for training HuggingFace models with the converted PISA dataset. This script is based on the official HuggingFace examples: https://github.com/huggingface/transformers/blob/main/examples/legacy/seq2seq/finetune_trainer.py

run_inference.py: run evaluation on the test set of PISA and return RMSE, MAE, Missing_Rate.

An example script of training/evaluation is given in example_script.sh

```bash
python3 run_hf.py \
    --model_name_or_path google_pegasus-xsum\  # can be replaced with other models
    --do_train \
    --do_eval \  
    --seed=66 \  # can select differenet seeds
    --save_total_limit=1 \
    --train_file SG/train.json \  # need to warp the PISA data (txt files) into json format via prepare_hf.py
    --validation_file SG/val.json \
    --output_dir SG_Pretrained_Pegasus \
    --per_device_train_batch_size=16 \
    --overwrite_output_dir \
    --predict_with_generate

python3 /run_inference.py \
      -t SG \  # path to the test set of PISA
      -m SG_Pretrained_Pegasus \
      -s G_Pretrained_Pegasus_pred \
      -d SG \
      --model_name pegasus \
      -b 16

```

