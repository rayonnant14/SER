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
        y = torch.from_numpy(self.data["y"][idx]).argmax()

        if "x" in self.use_keys and "x_opensmile" in self.use_keys:
            x = torch.from_numpy(self.data["x"][idx].transpose())
            x_opensmile = torch.from_numpy(self.data["x_opensmile"][idx])
            return x, x_opensmile, y
        elif "x" in self.use_keys and "x_lm" in self.use_keys and "x_asr" in self.use_keys:
            x = torch.from_numpy(self.data["x"][idx].transpose())
            x_asr = torch.from_numpy(self.data["x_asr"][idx])
            x_lm = torch.from_numpy(self.data["x_lm"][idx])
            return x, x_asr, x_lm, y
        elif "x" in self.use_keys and "x_asr" in self.use_keys:
            x = torch.from_numpy(self.data["x"][idx].transpose())
            x_asr = torch.from_numpy(self.data["x_asr"][idx])
            return x, x_asr, y
        elif "x" in self.use_keys and "x_lm" in self.use_keys:
            x = torch.from_numpy(self.data["x"][idx].transpose())
            x_lm = torch.from_numpy(self.data["x_lm"][idx])
            return x, x_lm, y
        elif "x_opensmile" in self.use_keys:
            x_opensmile = torch.from_numpy(self.data["x_opensmile"][idx])
            return x_opensmile, y
        elif "x_asr" in self.use_keys:
            x_opensmile = torch.from_numpy(self.data["x_asr"][idx])
            return x_opensmile, y
        elif "x_lm" in self.use_keys:
            x_opensmile = torch.from_numpy(self.data["x_lm"][idx])
            return x_opensmile, y
        else:
            x = torch.from_numpy(self.data["x"][idx].transpose())
            return x, y


def load_ser_dataset(dataset_path, use_keys=["x", "y"]):
    dataset = SERDataset(dataset_path, use_keys)
    return dataset