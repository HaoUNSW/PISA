import numpy as np
import os
import datetime


def add_days(start_day, delta_days):
    date_1 = datetime.datetime.strptime(start_day, '%B %d, %Y')
    end_date = date_1 + datetime.timedelta(days=delta_days)
    out = end_date.strftime('%B %d, %Y')

    return out


def generate_numerical(raw_folder, save_path, mode="test", obs_length=15):
    raw_data = np.load(os.path.join(raw_folder, "sg_raw_" + mode + ".npy"))
    data_x = []
    data_y = []
    for i in range(len(raw_data)):
        data = raw_data[i]
        number_of_instance = len(data) - obs_length
        for j in range(number_of_instance):
            y = data[obs_length + j]
            x = data[j: obs_length + j]
            data_x.append(x)
            data_y.append(y)

    data_x = np.reshape(data_x, [-1, obs_length])
    np.save(os.path.join(save_path, mode + "_" + str(obs_length) + "_x.npy"), data_x)
    data_y = np.reshape(data_y, [-1, 1])
    np.save(os.path.join(save_path, mode + "_" + str(obs_length) + "_y.npy"), data_y)


def output_sentence(target_usage):
    out = f"There will be {target_usage} visitors."

    return out


def input_sentence(usage, poi_id, start_date, obs_length):
    end_day = add_days(start_date, delta_days=obs_length - 1)
    prediction_day = add_days(start_date, delta_days=obs_length)
    start_week_day = datetime.datetime.strptime(start_date, '%B %d, %Y').strftime('%A')
    end_week_day = datetime.datetime.strptime(end_day, '%B %d, %Y').strftime('%A')
    prediction_week_day = datetime.datetime.strptime(prediction_day, '%B %d, %Y').strftime('%A')
    num_visits_string = ', '.join(map(str, usage))
    out = f"From {start_date}, {start_week_day} to {end_day}, {end_week_day}, there were {num_visits_string} people visiting POI {poi_id} on each day. How many people will visit POI {poi_id} on {prediction_day}, {prediction_week_day}?"

    return out


def generate_prompt(raw_folder, save_path, mode="train", obs_length=15, first_day="January 1, 2012"):
    raw_data = np.load(os.path.join(raw_folder, "sg_raw_" + mode + ".npy"))
    data_x_prompt = []
    data_y_prompt = []
    for i in range(len(raw_data)):
        data = raw_data[i]
        number_of_instance = len(data) - obs_length
        for j in range(number_of_instance):
            start_day = add_days(first_day, j)
            y = data[obs_length + j]
            x = data[j: obs_length + j]
            data_y_prompt.append(output_sentence(y))
            data_x_prompt.append(input_sentence(x, i+1, start_day, obs_length))

    with open(os.path.join(save_path, mode + "_15_x_prompt.txt"), "w") as f:
        for i in data_x_prompt:
            f.write(i + "\n")
        f.close()

    with open(os.path.join(save_path, mode + "_y_prompt.txt"), "w") as f:
        for i in data_y_prompt:
            f.write(i + "\n")
        f.close()
