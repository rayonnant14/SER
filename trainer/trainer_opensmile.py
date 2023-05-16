import torch

from trainer import TrainerClassification
from utils.metrics import accuracy


class TrainerOpenSmile(TrainerClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save_best_model(self, model, val_uar, fold):
        if val_uar > self.best_uar:
            self.best_uar = val_uar
            torch.save(
                model.state_dict(),
                self.save_path + "opensmile_" + str(fold) + ".pth",
            )
