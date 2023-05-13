import numpy as np
import torch
from torch.utils.data import Dataset


class TIMNetDataset(Dataset):
    def __init__(self, data_file, transform=None, data_size=None):
        """
        Args:
            data_file (numpy save): file with all data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        data = np.load(data_file, allow_pickle=True).item()
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data["x"])

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data["x"][idx].transpose())
        y = torch.from_numpy(self.data["y"][idx]).argmax()
        return x, y

def load_timnet_dataset(dataset_path):
    dataset = TIMNetDataset(dataset_path)
    return dataset

