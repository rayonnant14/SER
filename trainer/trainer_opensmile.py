import torch

from trainer import TrainerClassification
from utils.metrics import accuracy


class TrainerOpenSmile(TrainerClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save_best_model(self, model, val_accuracy, fold):
        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            torch.save(
                model.state_dict(),
                self.save_path + "opensmile_" + str(fold) + ".pth",
            )
