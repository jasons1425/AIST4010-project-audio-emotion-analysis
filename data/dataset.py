import torch
from torch.utils.data import Dataset


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
