import numpy as np
import torch
from data import SERDataset


class SEROpenSmileDataset(SERDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data["x"][idx].transpose())
        x_opensmile = torch.from_numpy(
            self.data["x_opensmile"][idx]
        )
        y = torch.from_numpy(self.data["y"][idx]).argmax()
        return x, x_opensmile, y


def load_opensmile_dataset(dataset_path):
    dataset = SEROpenSmileDataset(dataset_path)
    return dataset
