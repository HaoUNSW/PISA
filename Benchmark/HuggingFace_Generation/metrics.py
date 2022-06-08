import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def np_evaluate(gt_output, pred_output):
    rmse = mean_squared_error(gt_output, pred_output, squared=False)
    mae = mean_absolute_error(gt_output, pred_output)

    return rmse, mae


def metric_with_missing_rate(gt_text, predicted_text, dataset):
    output_data = []
    gt_data = []
    missing_count = 0

    for i in range(len(gt_text)):
        predicted_line = predicted_text[i]
        gt_line = gt_text[i]
        try:
            if dataset == 'SG':
                out = int(predicted_line.split(" ")[3])
                gt_out = int(gt_line.split(" ")[3])
            else:
                out = int(predicted_line.split(" ")[4])
                gt_out = int(gt_line.split(" ")[4])
            output_data.append(out)
            gt_data.append(gt_out)
        except Exception:
            missing_count += 1

    output = np.reshape(output_data, [len(output_data), 1])
    gt_output = np.reshape(gt_data, [len(gt_data), 1])

    rmse, mae = np_evaluate(gt_output, output)
    missing_rate = missing_count / len(gt_text)

    return rmse, mae, missing_rate







