import torch

from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold

import numpy as np

from tqdm import tqdm
from trainer import Base


class EvaluatorClassification(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model_weights(self, model, fold):
        model_path = (
            self.save_path + model.get_name() + "_" + str(fold) + ".pth"
        )
        model.load_state_dict(torch.load(model_path))

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

                if self.with_pca:
                    _, val_loader = self.process_dataloader(
                        train_sampler, val_sampler
                    )
                else:
                    val_loader = DataLoader(
                        self.dataset,
                        batch_size=self.batch_size,
                        sampler=val_sampler,
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
