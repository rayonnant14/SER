import numpy as np
import torch
from torch.utils.data import Dataset


class SERDataset(Dataset):
    def __init__(self, data_file, use_keys, transform=None):
        """
        Args:
            data_file (numpy save): file with all data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        data = np.load(data_file, allow_pickle=True).item()
        self.data = data
        self.transform = transform
        self.use_keys = use_keys

    def __len__(self):
        return len(self.data["y"])

    def __getitem__(self, idx):
        sample = []
        for key in self.use_keys:
            try:
                value = torch.from_numpy(self.data[key][idx])
            except:
                value = torch.tensor(self.data[key][idx])
            sample.append(value)
        return tuple(sample)


def load_ser_dataset(dataset_path, use_keys=["x", "y"]):
    dataset = SERDataset(dataset_path, use_keys)
    return dataset