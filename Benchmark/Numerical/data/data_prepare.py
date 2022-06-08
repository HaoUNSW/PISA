from datetime import datetime, timedelta
import numpy as np
import os


def process_date(date_end_sting, length=16):
    # obs 15 days, predict 1 day
    output_xy_mark = np.ones((length, 3))
    pred_day = datetime.strptime(date_end_sting, '%B %d, %Y, %A ')
    for i in range(length):
        idx = length - 1 - i
        current_day = pred_day - timedelta(days=i)
        output_xy_mark[idx, 0] = current_day.month
        output_xy_mark[idx, 1] = current_day.day
        output_xy_mark[idx, 2] = current_day.weekday()

    return output_xy_mark


def text_to_informer(input_file_path, output_file_path, data_set, seq_len=15, label_len=7, pred_len=1):
    s_begin = 0
    s_end = s_begin + seq_len
    r_begin = s_end - label_len

    with open(output_file_path, "r") as fo:
        output_text_file = fo.readlines()

    with open(input_file_path, "r") as fi:
        input_text_file = fi.readlines()

    seq_x = []
    seq_y = []
    seq_x_mark = []
    seq_y_mark = []

    if len(output_text_file) == len(input_text_file):
        for i in range(len(input_text_file)):
            output_line = output_text_file[i]
            input_line = input_text_file[i]
            if data_set == "SG":
                out_y = int(output_line.split(" ")[3])
            else:
                out_y = int(output_line.split(" ")[4])
            end_day = input_line.split("on ")[-1].replace("?", "")
            # print(end_day)
            xy_mark = process_date(end_day, length=seq_len + pred_len)
            # print(xy_mark)
            if data_set == "SG":
                b = input_line.split("there were")[1]
                c = b.split("people")[0]
                d = c.split(",")
                d = [int(x) for x in d]
            elif data_set == "CT":
                b = input_line.split("was")[1]
                c = b.split("degree")[0]
                d = c.split(",")
                d = [int(x) for x in d]
            else:
                b = input_line.split("consumed")[1]
                c = b.split("kWh")[0]
                d = c.split(",")
                d = [int(x) for x in d]

            d.append(out_y)
            input_xy = np.array(d).reshape([1, -1])
            seq_x.append(input_xy[:, s_begin:s_end])
            seq_y.append(input_xy[:, r_begin:])
            seq_x_mark.append(xy_mark[s_begin:s_end, :].reshape([1, seq_len, 3]))
            seq_y_mark.append(xy_mark[r_begin:, :].reshape([1, label_len + pred_len, 3]))

    seq_x = np.concatenate(seq_x, axis=0).reshape([-1, seq_len, 1])
    seq_y = np.concatenate(seq_y, axis=0).reshape([-1, label_len + pred_len, 1])
    seq_x_mark = np.concatenate(seq_x_mark, axis=0)
    seq_y_mark = np.concatenate(seq_y_mark, axis=0)

    print(np.shape(seq_x))
    print(np.shape(seq_y))
    print(np.shape(seq_x_mark))
    print(np.shape(seq_y_mark))
    return seq_x, seq_y, seq_x_mark, seq_y_mark


def combine(prompt_file_folder, save_path, dataset, seq_len=15, label_len=7, pred_len=1):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print("train")
    train_in_path = os.path.join(prompt_file_folder, 'train_15_x_prompt.txt')
    train_out_path = os.path.join(prompt_file_folder, 'train_15_y_prompt.txt')
    trainseq_x, trainseq_y, trainseq_x_mark, trainseq_y_mark = text_to_informer(train_in_path, train_out_path,
                                                                                data_set=dataset,
                                                                                seq_len=seq_len, label_len=label_len,
                                                                                pred_len=pred_len)
    print("val")
    val_in_path = os.path.join(prompt_file_folder, 'val_15_x_prompt.txt')
    val_out_path = os.path.join(prompt_file_folder, 'val_15_y_prompt.txt')
    valseq_x, valseq_y, valseq_x_mark, valseq_y_mark = text_to_informer(val_in_path, val_out_path, data_set=dataset,
                                                                        seq_len=seq_len, label_len=label_len,
                                                                        pred_len=pred_len)
    print("test")
    test_in_path = os.path.join(prompt_file_folder, 'test_15_x_prompt.txt')
    test_out_path = os.path.join(prompt_file_folder, 'test_15_y_prompt.txt')
    testseq_x, testseq_y, testseq_x_mark, testseq_y_mark = text_to_informer(test_in_path, test_out_path,
                                                                            data_set=dataset,
                                                                            seq_len=seq_len, label_len=label_len,
                                                                            pred_len=pred_len)
    seq_x = np.concatenate((trainseq_x, valseq_x, testseq_x), axis=0)
    seq_y = np.concatenate((trainseq_y, valseq_y, testseq_y), axis=0)
    seq_x_mark = np.concatenate((trainseq_x_mark, valseq_x_mark, testseq_x_mark), axis=0)
    seq_y_mark = np.concatenate((trainseq_y_mark, valseq_y_mark, testseq_y_mark), axis=0)
    np.save(os.path.join(save_path, "seq_x.npy"), seq_x)
    np.save(os.path.join(save_path, "seq_y.npy"), seq_y)
    np.save(os.path.join(save_path, "seq_x_mark.npy"), seq_x_mark)
    np.save(os.path.join(save_path, "seq_y_mark.npy"), seq_y_mark)


if __name__ == "__main__":
    combine("prompt_path",
            "processed_data_path",
            dataset="CT",
            seq_len=15,
            label_len=7,
            pred_len=1)
