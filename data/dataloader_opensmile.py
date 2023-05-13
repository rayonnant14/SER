import numpy as np
import torch
from .dataloader_timnet import TIMNetDataset


class OpenSmileDataset(TIMNetDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        x_opensmile = torch.from_numpy(self.data["x_opensmile"][idx])
        y = torch.from_numpy(self.data["y"][idx]).argmax()
        return x_opensmile, y


def load_opensmile_dataset(dataset_path):
    dataset = OpenSmileDataset(dataset_path)
    return dataset
