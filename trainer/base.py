from abc import ABC

import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from data.datasets import DATASETS

from sklearn.metrics import recall_score, accuracy_score, classification_report
from utils.misc import check_if_exist


class Base(ABC):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        dataset_name: str,
        model_class: nn.Module,
        batch_size: int,
        save_path: str,
        with_pca=False,
        pca_components=100,
        device=None,
    ):
        self.dataset = dataset
        self.dataset_description = DATASETS[dataset_name]
        self.model_class = model_class
        self.batch_size = batch_size
        self.device = device
        self.n_splits = 10
        self.random_state = self.dataset_description["seed"]
        self.dilations = self.dataset_description["dilations"]
        self.with_pca = with_pca
        self.pca_components = pca_components
        check_if_exist(save_path + "/" + dataset_name)
        self.save_path = save_path + "/" + dataset_name + "/"

    def load_model(self):
        model = self.model_class(
            class_num=self.dataset_description["num_classes"],
            dilations=self.dilations
        )
        return model

    def train_mode_on(self, model):
        model.train()

    def eval_mode_on(self, model):
        model.eval()

    def validation_step(self, model, batch):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        out = model.forward(images)
        return labels, out

    def process_dataloader(self, train_idx, val_idx):
        train_set = Subset(self.dataset, train_idx)
        val_set = Subset(self.dataset, val_idx)
        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
        )
        return train_loader, val_loader

    @torch.no_grad()
    def evaluate(self, model, val_loader, report=False):
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
        if report:
            report = classification_report(
                labels_all,
                preds_all,
                labels=np.arange(
                    0, len(self.dataset_description["target_names"]), 1
                ),
                target_names=self.dataset_description["target_names"],
            )
            metrics = {"WAR": war, "UAR": uar, "report": report}
        else:
            metrics = {"WAR": war, "UAR": uar}
        return metrics
