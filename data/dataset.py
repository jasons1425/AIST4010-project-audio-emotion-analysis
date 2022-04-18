import torch
from torch.utils.data import Dataset
import numpy as np
import re
import glob
import os


class ImageDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        super(ImageDataset, self).__init__()
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(self.data[idx])
        return sample, self.labels[idx]

    def get_data(self, idx=None):
        if idx:
            return self.data[idx]
        return self.data


class LazyWavDataset(Dataset):
    def __init__(self, files, labels,
                 re_pattern=r".*\\([^\\\.]*)\.npy", transform=None, extension='.npy',
                 pad_len=661500, sample_size=8500):
        super().__init__()
        self.files = files
        self.ids = [re.match(re_pattern, fp).group(1) for fp in self.files]
        self.labels = torch.from_numpy(labels)
        if sample_size:
            self.files = self.files[:8500]
            self.labels = self.labels[:8500]
        self.transform = transform
        self.pad_len = pad_len

    def __getitem__(self, item):
        file = self.files[item]
        wav = np.load(file)
        pad_len = self.pad_len
        if pad_len:
            if len(wav) >= pad_len:
                wav = wav[:pad_len]
            else:
                wav = np.pad(wav, (0, pad_len - len(wav)),
                             mode='constant', constant_values=(0, 0))
        file = torch.from_numpy(wav)
        label = self.labels[item]
        if self.transform:
            file = self.transform(file)
        return file, label

    def get_item_with_id(self, item):
        file = self.files[item]
        song_id = self.ids[item]
        wav = np.load(file)
        pad_len = self.pad_len
        if pad_len:
            if len(wav) >= pad_len:
                wav = wav[:pad_len]
            else:
                wav = np.pad(wav, (0, pad_len - len(wav)),
                             mode='constant', constant_values=(0, 0))
        file = torch.from_numpy(wav)
        label = self.labels[item]
        if self.transform:
            file = self.transform(file)
        return song_id, file, label

    def __len__(self):
        return len(self.files)
