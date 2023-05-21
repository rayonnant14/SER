import numpy as np
import torch
from torch.utils.data import Dataset


class SERDatasetPCA(Dataset):
    def __init__(self, data, use_keys):
        """
        Args:
            data_file (numpy save): file with all data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.use_keys = use_keys

    def __len__(self):
        return len(self.data["y"])
    
    def __getitem__(self, idx):
        y = self.data["y"][idx]

        if "x" in self.use_keys and "x_additional" in self.use_keys:
            x = self.data["x"][idx]
            x_additional = torch.from_numpy(self.data["x_additional"][idx])
            return x, x_additional, y
        else:
            x_additional = torch.from_numpy(self.data["x_additional"][idx])
            return x_additional, y


def load_ser_pca_dataset(data, use_keys=["x", "y"]):
    dataset = SERDatasetPCA(data, use_keys)
    return dataset
