import torch

from trainer import TrainerClassification
import numpy as np


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
        return labels, out

    def save_best_model(self, model, val_uar, fold):
        if val_uar > self.best_uar:
            self.best_uar = val_uar
            torch.save(
                model.state_dict(),
                self.save_path + "two_branches_" + str(fold) + ".pth",
            )

    def load_model(self):
        model = self.model_class(
            class_num=self.dataset_description["num_classes"],
            with_pca=self.with_pca,
        )
        return model

    def process_dataloader(self, train_loader, val_loader):
        train_loader, val_loader = self.apply_pca(train_loader, val_loader)
        return train_loader, val_loader
