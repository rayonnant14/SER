from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold

from utils.metrics import accuracy
from utils.misc import check_if_exist

from data.datasets import DATASETS

from tqdm import tqdm


class TrainerClassification(ABC):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        dataset_name: str,
        model_class: nn.Module,
        batch_size: int,
        optimizer_func,
        optimizer_parameters,
        criterion,
        num_epochs: int,
        save_path: str,
        device=None,
    ):
        self.dataset = dataset
        self.dataset_description = DATASETS[dataset_name]
        self.model_class = model_class
        self.batch_size = batch_size
        self.device = device
        self.optimizer_func = optimizer_func
        self.optimizer_parameters = optimizer_parameters
        self.criterion = criterion
        self.epochs = num_epochs
        self.best_accuracy = 0.0
        self.n_splits = 10
        self.random_state = 42

        check_if_exist(save_path + "/" + dataset_name)
        self.save_path = save_path + "/" + dataset_name + "/"

    def load_model(self):
        model = self.model_class(
            class_num=self.dataset_description["num_classes"]
        )
        return model

    def train_mode_on(self, model):
        model.train()

    def eval_mode_on(self, model):
        model.eval()

    def training_step(self, model, batch):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        out = model.forward(images)
        loss = self.criterion(out, labels)
        loss.backward()
        return loss

    def validation_step(self, model, batch):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        out = model.forward(images)
        acc = accuracy(out, labels)
        return {"val_acc": acc}

    def validation_epoch_end(self, outputs):
        batch_accs = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {"val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], train_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result["train_loss"], result["val_acc"]
            )
        )

    def save_best_model(self, model, val_accuracy, fold):
        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            torch.save(
                model.state_dict(),
                self.save_path + "timnet_" + str(fold) + ".pth",
            )

    @torch.no_grad()
    def evaluate(self, model, val_loader):
        self.eval_mode_on(model)
        outputs = [self.validation_step(model, batch) for batch in val_loader]
        return self.validation_epoch_end(outputs)

    def fit(self):
        splits = KFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        with tqdm(total=self.n_splits) as pbar:
            for fold, (train_idx, val_idx) in enumerate(
                splits.split(np.arange(len(self.dataset)))
            ):
                print(f"Process fold {fold}")
                self.best_accuracy = 0.0
                history = []
                train_sampler = SubsetRandomSampler(train_idx)
                val_sampler = SubsetRandomSampler(val_idx)
                train_loader = DataLoader(
                    self.dataset,
                    batch_size=self.batch_size,
                    sampler=train_sampler,
                )
                val_loader = DataLoader(
                    self.dataset,
                    batch_size=self.batch_size,
                    sampler=val_sampler,
                )
                model = self.load_model()
                model.to(self.device)

                optimizer = self.optimizer_func(
                    model.parameters(), **self.optimizer_parameters
                )
                for epoch in range(self.epochs):
                    self.train_mode_on(model)
                    train_losses = []
                    for batch in train_loader:
                        loss = self.training_step(model, batch)
                        optimizer.step()
                        optimizer.zero_grad()
                        train_losses.append(loss)

                    result = self.evaluate(model, val_loader)
                    result["train_loss"] = (
                        torch.stack(train_losses).mean().item()
                    )
                    self.epoch_end(epoch, result)
                    self.save_best_model(model, result["val_acc"], fold)
                    history.append(result)
                pbar.update(1)
                # yield history
