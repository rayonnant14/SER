import torch

from trainer import TrainerClassification
from utils.metrics import accuracy


class TrainerTwoBranches(TrainerClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, model, batch):
        x, x_opensmile, labels = batch
        x, x_opensmile, labels = (
            x.to(self.device),
            x_opensmile.to(self.device),
            labels.to(self.device),
        )
        out = model.forward(x, x_opensmile)
        loss = self.criterion(out, labels)
        loss.backward()
        return loss

    def validation_step(self, model, batch):
        x, x_opensmile, labels = batch
        x, x_opensmile, labels = (
            x.to(self.device),
            x_opensmile.to(self.device),
            labels.to(self.device),
        )
        out = model.forward(x, x_opensmile)
        acc = accuracy(out, labels)
        return {"val_acc": acc}

    def save_best_model(self, model, val_accuracy, fold):
        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            torch.save(
                model.state_dict(),
                self.save_path + "two_branches_" + str(fold) + ".pth",
            )
