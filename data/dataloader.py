import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class SERDataset(Dataset):
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


# def split_dataset(dataset_path):
#     dataset = SERDataset(dataset_path)
#     val_size = int(len(dataset) * 0.2)
#     train_size = len(dataset) - val_size

#     train_data, val_data = random_split(dataset, [train_size,val_size])
#     return train_data, val_data


def load_dataset(dataset_path):
    dataset = SERDataset(dataset_path)
    return dataset
    # train_data, val_data = split_dataset(dataset_path)
    # train_dataloader = DataLoader(train_data, batch_size, shuffle = True, num_workers = 4)
    # val_dataloader = DataLoader(val_data, batch_size, num_workers = 4)
    # return train_dataloader, val_dataloader
