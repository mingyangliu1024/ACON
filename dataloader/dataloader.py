"""
@author: Mingyang Liu
@contact: mingyangliu1024@gmail.com
"""

import os
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class Load_Dataset(Dataset):
    def __init__(self, dataset):
        super(Load_Dataset, self).__init__()

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train)
            y_train = torch.from_numpy(y_train).long()

        # (channel, length)
        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)
        
        if X_train.shape.index(min(X_train.shape[1], X_train.shape[2])) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        self.x_data = X_train
        self.y_data = y_train

        self.num_channels = X_train.shape[1]

        # normalize
        data_mean = torch.mean(X_train, dim=(0, 2))
        data_std = torch.std(X_train, dim=(0, 2))
        self.transform = transforms.Normalize(mean=data_mean, std=data_std)
        self.len = X_train.shape[0]

    def __getitem__(self, index):
        if self.transform is not None:
            output = self.transform(self.x_data[index].view(self.num_channels, -1, 1))
            self.x_data[index] = output.view(self.x_data[index].shape)

        return self.x_data[index].float(), self.y_data[index].long()

    def __len__(self):
        return self.len


def data_generator(data_path, domain_id, args):
    # loading path
    train_dataset = torch.load(os.path.join(data_path, "train_" + domain_id + ".pt"))
    test_dataset = torch.load(os.path.join(data_path, "test_" + domain_id + ".pt"))

    # Loading datasets
    train_dataset = Load_Dataset(train_dataset)
    test_dataset = Load_Dataset(test_dataset)
    batch_size = args.bs
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=args.shuffle, drop_last=True, num_workers=args.num_workers)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=True, num_workers=args.num_workers)
    return train_loader, test_loader
