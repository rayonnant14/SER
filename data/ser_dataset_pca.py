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

        if "x" in self.use_keys and "x_opensmile" in self.use_keys:
            x = self.data["x"][idx]
            x_opensmile = torch.from_numpy(self.data["x_opensmile"][idx])
            return x, x_opensmile, y
        else:
            x_opensmile = torch.from_numpy(self.data["x_opensmile"][idx])
            return x_opensmile, y


def load_ser_pca_dataset(data, use_keys=["x", "y"]):
    dataset = SERDatasetPCA(data, use_keys)
    return dataset
