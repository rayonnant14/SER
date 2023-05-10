import torch
import torch.nn as nn

from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from data.datasets import DATASETS

from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, accuracy_score
import numpy as np

from tqdm import tqdm


class EvaluatorClassification:
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        dataset_name: str,
        model_class: nn.Module,
        batch_size: int,
        save_path: str,
        device=None,
    ):
        self.dataset = dataset
        self.dataset_description = DATASETS[dataset_name]
        self.model_class = model_class
        self.batch_size = batch_size
        self.device = device
        self.n_splits = 10
        self.random_state = 42
        self.save_path = save_path + "/" + dataset_name + "/"

    def load_model(self):
        model = self.model_class(
            class_num=self.dataset_description["num_classes"]
        )
        return model

    def load_model_weights(self, model, fold):
        model_path = self.save_path + "timnet_" + str(fold) + ".pth"
        print(model_path)
        model.load_state_dict(torch.load(model_path))

    def eval_mode_on(self, model):
        model.eval()

    def infer_model(self, model, batch):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        out = model.forward(images)
        return out

    @torch.no_grad()
    def evaluate_model(self, model, val_loader):
        outputs = [
            torch.argmax(self.infer_model(model, batch), dim=1)
            .cpu()
            .detach()
            .numpy()
            for batch in val_loader
        ]
        outputs = np.concatenate(outputs, axis=0)
        targets = [batch[1].cpu().detach().numpy() for batch in val_loader]
        targets = np.concatenate(targets, axis=0)
        print(
            classification_report(
                targets,
                outputs,
                target_names=self.dataset_description["target_names"],
            )
        )
        war = recall_score(targets, outputs, average="weighted")
        uar = accuracy_score(targets, outputs)
        return {'WAR': war, 'UAR': uar}

    def evaluate(self):
        splits = KFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        average_WAR = 0.0
        average_UAR = 0.0
        with tqdm(total=self.n_splits) as pbar:
            for fold, (_, val_idx) in enumerate(
                splits.split(np.arange(len(self.dataset)))
            ):
                print(f"Process fold {fold}")
                val_sampler = SubsetRandomSampler(val_idx)
                val_loader = DataLoader(
                    self.dataset,
                    batch_size=self.batch_size,
                    sampler=val_sampler,
                )
                model = self.load_model()
                self.load_model_weights(model, fold)
                model.to(self.device)
                self.eval_mode_on(model)
                metrics = self.evaluate_model(model, val_loader)
                average_WAR += metrics['WAR']
                average_UAR += metrics['UAR']
                pbar.update(1)
                # yield history
        average_WAR /= self.n_splits
        average_UAR /= self.n_splits
        print(f'average_WAR: {average_WAR}, average_UAR: {average_UAR}')
