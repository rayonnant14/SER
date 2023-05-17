from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from data import load_pca_dataset
from sklearn.decomposition import PCA

from utils.metrics import accuracy
from sklearn.metrics import recall_score, accuracy_score
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
        with_pca=False,
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
        self.best_uar = 0.0
        self.n_splits = 10
        self.random_state = 42
        self.with_pca = with_pca
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
        return labels, out

    def process_dataloader(self, train_loader, val_loader):
        return train_loader, val_loader

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], train_loss: {:.4f}, val_UAR: {:.4f}, val_WAR: {:.4f}".format(
                epoch, result["train_loss"], result["UAR"], result["WAR"]
            )
        )

    def save_best_model(self, model, val_uar, fold):
        if val_uar > self.best_uar:
            self.best_uar = val_uar
            torch.save(
                model.state_dict(),
                self.save_path + "timnet_" + str(fold) + ".pth",
            )

    @torch.no_grad()
    def evaluate(self, model, val_loader):
        self.eval_mode_on(model)
        labels_all = []
        preds_all = []
        for batch in val_loader:
            labels, preds = self.validation_step(model, batch)
            labels = labels.cpu().detach().numpy()
            preds = torch.argmax(preds, dim=1).cpu().detach().numpy()
            labels_all.append(labels)
            preds_all.append(preds)
        labels_all = np.concatenate(labels_all, axis=0)
        preds_all = np.concatenate(preds_all, axis=0)
        uar = recall_score(labels_all, preds_all, average="macro")
        war = accuracy_score(labels_all, preds_all)
        metrics = {"WAR": war, "UAR": uar}
        return metrics
        # return self.validation_epoch_end(outputs)

    def fit(self):
        splits = KFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        with tqdm(total=self.n_splits) as pbar:
            for fold, (train_idx, val_idx) in enumerate(
                splits.split(np.arange(len(self.dataset)))
            ):
                print(f"Process fold {fold}")
                self.best_uar = 0.0
                history = []
                train_sampler = SubsetRandomSampler(train_idx)
                val_sampler = SubsetRandomSampler(val_idx)
                if self.with_pca:
                    temp_batch_size = len(self.dataset)
                else:
                    temp_batch_size = self.batch_size
                train_loader = DataLoader(
                    self.dataset,
                    batch_size=temp_batch_size,
                    sampler=train_sampler,
                )
                val_loader = DataLoader(
                    self.dataset,
                    batch_size=temp_batch_size,
                    sampler=val_sampler,
                )
                train_loader, val_loader = self.process_dataloader(
                    train_loader, val_loader
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
                    self.save_best_model(model, result["UAR"], fold)
                    history.append(result)
                pbar.update(1)
                # yield history

    def apply_pca(self, train_loader, val_loader):
        x_train, x_opensmile_train, y_train = next(iter(train_loader))
        x_val, x_opensmile_val, y_val = next(iter(val_loader))

        x_opensmile_train = x_opensmile_train.view(-1, 988).numpy()
        x_opensmile_val = x_opensmile_val.view(-1, 988).numpy()
        pca = PCA(n_components=100)
        x_train_pca = pca.fit_transform(x_opensmile_train)
        x_val_pca = pca.transform(x_opensmile_val)

        train_pca = {"x": x_train, "x_opensmile_pca": x_train_pca, "y": y_train}
        val_pca = {"x": x_val, "x_opensmile_pca": x_val_pca, "y": y_val}

        train_dataset_pca = load_pca_dataset(train_pca)
        val_dataset_pca = load_pca_dataset(val_pca)
        train_loader_pca = DataLoader(
            train_dataset_pca,
            batch_size=self.batch_size,
        )
        val_loader_pca = DataLoader(
            val_dataset_pca,
            batch_size=self.batch_size,
        )
        return train_loader_pca, val_loader_pca
