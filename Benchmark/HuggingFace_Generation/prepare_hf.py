import pandas as pd
import os
import csv
import jsonlines


def clean(x):
    x = x.replace(",", "")
    return x


def to_jsonl(src_file, dst_file):
    all_files = os.listdir(src_file)
    if not os.path.exists(dst_file):
        os.mkdir(dst_file)
    for f in all_files:
        if "val_15_x_prompt" in f:
            val_in_list = open(os.path.join(src_file, f)).readlines()
            val_in_list = [line.replace("\n", "") for line in val_in_list]
        elif "val_15_y_prompt" in f:
            val_out_list = open(os.path.join(src_file, f)).readlines()
            # val_out_list = [line.replace("\n", ",") for line in val_out_list]
            val_out_list = [line.replace("\n", "") for line in val_out_list]
        elif "train_15_x_prompt" in f:
            train_in_list = open(os.path.join(src_file, f)).readlines()
            train_in_list = [line.replace("\n", "") for line in train_in_list]
        elif "train_15_y_prompt" in f:
            train_out_list = open(os.path.join(src_file, f)).readlines()
            train_out_list = [line.replace("\n", "") for line in train_out_list]

    val_items = []
    train_items = []

    for i in range(len(val_in_list)):
        val_items.append({"text": val_in_list[i], "summary": val_out_list[i]})
    for i in range(len(train_in_list)):
        train_items.append({"text": train_in_list[i], "summary": train_out_list[i]})
    with jsonlines.open(os.path.join(dst_file, "val.json"), 'w') as writer:
        writer.write_all(val_items)
    with jsonlines.open(os.path.join(dst_file, "train.json"), 'w') as writer:
        writer.write_all(train_items)





