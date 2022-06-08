from utils.tools import StandardScaler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import os


class FormerData:
    def __init__(self, cfg_params):
        cfg_params.copyAttrib(self)
        seq_x = np.load(os.path.join(self.seq_data_path, "seq_x.npy"))
        seq_y = np.load(os.path.join(self.seq_data_path, "seq_y.npy"))
        self.seq_x_mark = np.load(os.path.join(self.seq_data_path, "seq_x_mark.npy"))
        self.seq_y_mark = np.load(os.path.join(self.seq_data_path, "seq_y_mark.npy"))
        self.scaler = StandardScaler()
        print(np.shape(seq_x), np.shape(seq_y[:, -1, :]))
        self.scaler.fit(np.concatenate((seq_x, seq_y[:, -1, :].reshape(-1, 1, 1)), axis=1))
        self.seq_x = self.scaler.transform(seq_x)
        self.seq_y = self.scaler.transform(seq_y)

    def get_dataset(self, args, flag="train"):

        if flag == "train":
            dataset = TensorDataset(torch.from_numpy(self.seq_x[:self.train_num]),
                                    torch.from_numpy(self.seq_y[:self.train_num]),
                                    torch.from_numpy(self.seq_x_mark[:self.train_num]),
                                    torch.from_numpy(self.seq_y_mark[:self.train_num]))
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        elif flag == "val":
            dataset = TensorDataset(torch.from_numpy(self.seq_x[self.train_num: self.train_num + self.val_num]),
                                    torch.from_numpy(self.seq_y[self.train_num: self.train_num + self.val_num]),
                                    torch.from_numpy(self.seq_x_mark[self.train_num: self.train_num + self.val_num]),
                                    torch.from_numpy(self.seq_y_mark[self.train_num: self.train_num + self.val_num]))
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        elif flag == "test":
            dataset = TensorDataset(torch.from_numpy(self.seq_x[-self.test_num:]),
                                    torch.from_numpy(self.seq_y[-self.test_num:]),
                                    torch.from_numpy(self.seq_x_mark[-self.test_num:]),
                                    torch.from_numpy(self.seq_y_mark[-self.test_num:]))
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        return self.scaler, data_loader

