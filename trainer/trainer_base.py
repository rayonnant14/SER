from trainer import Base

import torch
import numpy as np

from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold

from tqdm import tqdm


class TrainerClassification(Base):
    def __init__(
        self,
        optimizer_func,
        optimizer_parameters,
        criterion,
        num_epochs: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.optimizer_func = optimizer_func
        self.optimizer_parameters = optimizer_parameters
        self.criterion = criterion
        self.epochs = num_epochs
        self.best_uar = 0.0

    def training_step(self, model, batch):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        out = model.forward(images)
        loss = self.criterion(out, labels)
        loss.backward()
        return loss

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], train_loss: {:.4f}, val_UAR: {:.4f}, val_WAR: {:.4f}".format(
                epoch, result["train_loss"], result["UAR"], result["WAR"]
            )
        )

    def save_best_model(self, model, val_uar, fold):
        if val_uar >= self.best_uar:
            self.best_uar = val_uar
            torch.save(
                model.state_dict(),
                self.save_path + model.get_name() + "_" + str(fold) + ".pth",
            )

    def load_model_weights(self, model, fold):
        model_path = (
            self.save_path + model.get_name() + "_" + str(fold) + ".pth"
        )
        model.load_state_dict(torch.load(model_path))

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
                train_loader, val_loader = self.process_dataloader(
                    train_sampler, val_sampler
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

    def predict(self):
        splits = KFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        average_WAR = 0.0
        average_UAR = 0.0
        with tqdm(total=self.n_splits) as pbar:
            for fold, (train_idx, val_idx) in enumerate(
                splits.split(np.arange(len(self.dataset)))
            ):
                print(f"Process fold {fold}")
                train_sampler = SubsetRandomSampler(train_idx)
                val_sampler = SubsetRandomSampler(val_idx)

                _, val_loader = self.process_dataloader(
                    train_sampler, val_sampler
                )
                model = self.load_model()
                self.load_model_weights(model, fold)
                self.eval_mode_on(model)
                model.to(self.device)
                metrics = self.evaluate(model, val_loader, report=True)
                print(metrics["report"])
                average_WAR += metrics["WAR"]
                average_UAR += metrics["UAR"]
                pbar.update(1)
                # yield history
        average_WAR /= self.n_splits
        average_UAR /= self.n_splits
        average_WAR_perc = average_WAR * 100.0
        average_UAR_perc = average_UAR * 100.0
        print(
            f"average_UAR: {average_UAR_perc:.2f} %, average_WAR: {average_WAR_perc:.2f} %",
        )
        return {
            "average_UAR": round(average_UAR_perc, 2),
            "average_WAR": round(average_WAR_perc, 2),
        }
        # yield history
