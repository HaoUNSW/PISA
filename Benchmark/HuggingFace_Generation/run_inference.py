from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import BlenderbotSmallForConditionalGeneration, AutoTokenizer
from transformers import PegasusForConditionalGeneration, ProphetNetForConditionalGeneration
from transformers import T5ForConditionalGeneration
from transformers import LEDForConditionalGeneration, PegasusTokenizer, BigBirdPegasusForConditionalGeneration
import argparse
import os
import torch
from tqdm import tqdm
from metrics import metric_with_missing_rate


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('-t', '--test_file',
                      default='SG',
                      type=str)
    args.add_argument('-m', '--model_path', default='results',
                      type=str)
    args.add_argument('--model_name', default='bart',
                      type=str)
    args.add_argument('-s', '--save_path', default='results',
                      type=str)
    args.add_argument('-b', '--batch_size', default=200,
                      type=int)
    args.add_argument('-d', '--dataset_name', default='SG',
                      type=str)
    return args.parse_args()


def get_max_length(folder_path, tokenizer):
    input_lines = []
    output_lines = []

    files = os.listdir(folder_path)
    for f in files:
        if f.endswith("x_prompt.txt"):
            input_lines.extend(open(os.path.join(folder_path, f), "r").readlines())
        elif f.endswith("y_prompt.txt"):
            output_lines.extend(open(os.path.join(folder_path, f), "r").readlines())
    in_leng = []
    out_leng = []
    for sent in input_lines:
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            # max_length=max_length,  # Pad & truncate all sentences.
            # padding='max_length',
            return_attention_mask=False,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        # print(encoded_dict['input_ids'].size())
        in_leng.append(encoded_dict['input_ids'].size(1))
    for sent in output_lines:
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            # max_length=max_length,  # Pad & truncate all sentences.
            # padding='max_length',
            return_attention_mask=False,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        out_leng.append(encoded_dict['input_ids'].size(1))

    print(max(in_leng), max(out_leng))

    return max(in_leng), max(out_leng)


def get_tokens(input_file, tokenizer, max_length):
    token_ids = []

    for sent in input_file:
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_length,  # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=False,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        token_ids.append(encoded_dict['input_ids'])
    # #
    token_ids = torch.cat(token_ids, dim=0)

    return token_ids


if __name__ == "__main__":

    args = get_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model_name == 'bart':
        model = BartForConditionalGeneration.from_pretrained(args.model_path)
        tokenizer = BartTokenizer.from_pretrained(args.model_path)
    elif args.model_name == 'blendersmall':
        model = BlenderbotSmallForConditionalGeneration.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    elif args.model_name == 'pegasus':
        model = PegasusForConditionalGeneration.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    elif args.model_name == 'prophet':
        model = ProphetNetForConditionalGeneration.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    elif args.model_name == 'led':
        model = LEDForConditionalGeneration.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    elif args.model_name == 'T5':
        model = T5ForConditionalGeneration.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    elif args.model_name == 'bigbird':
        model = BigBirdPegasusForConditionalGeneration.from_pretrained(args.model_path)
        tokenizer = PegasusTokenizer.from_pretrained(args.model_path)
    model.to(device)

    gt_file = os.path.join(args.test_file, "test_15_y_prompt.txt")
    test_in_file = os.path.join(args.test_file, "test_15_x_prompt.txt")
    max_len_in, max_len_out = get_max_length(args.test_file, tokenizer)

    gt_text = open(gt_file, "r").readlines()
    in_lines = open(test_in_file, "r").readlines()
    all_pred = []
    # batching
    num_batch = len(in_lines) // args.batch_size
    for i in tqdm(range(num_batch + 1)):
        start_idx = i * args.batch_size
        end_idx = (i + 1) * args.batch_size
        if end_idx < len(in_lines):
            in_text = in_lines[start_idx: end_idx]
        else:
            in_text = in_lines[start_idx:]

        tokens = get_tokens(in_text, tokenizer, max_len_in)
        summary_ids = model.generate(tokens.to(device), num_beams=4, max_length=max_len_out, early_stopping=True)
        predicted = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
        all_pred.extend(predicted)

    with open(os.path.join(args.save_path, "predicted.txt"), "w") as f:
        f.writelines([x + "\n" for x in all_pred])
        f.close()
    rmse, mae, ms = metric_with_missing_rate(gt_text, all_pred, args.dataset_name)
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"Missing Rate: {ms}")

