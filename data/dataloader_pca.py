import numpy as np
import torch
from torch.utils.data import Dataset


class PCADataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (dict): dict with all data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data

    def __len__(self):
        return len(self.data["x_opensmile_pca"])

    def __getitem__(self, idx):
        x = self.data["x"][idx]
        x_opensmile_pca = torch.from_numpy(self.data["x_opensmile_pca"][idx])
        y = self.data["y"][idx]
        return x, x_opensmile_pca, y

class OpenSmilePCADataset(PCADataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        x_opensmile_pca = torch.from_numpy(self.data["x_opensmile_pca"][idx])
        y = self.data["y"][idx]
        return x_opensmile_pca, y
    

def load_pca_dataset(dataset):
    dataset = PCADataset(dataset)
    return dataset

def load_opensmile_pca_dataset(dataset):
    dataset = OpenSmilePCADataset(dataset)
    return dataset
