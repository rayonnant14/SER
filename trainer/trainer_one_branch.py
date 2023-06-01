from torch.utils.data import DataLoader

from trainer import TrainerClassification
from utils import apply_pca


class TrainerOneBranch(TrainerClassification):
    def __init__(
        self, features_num, with_pca=False, pca_components=100, 
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.features_num = features_num
        self.with_pca = with_pca
        self.pca_components = pca_components

    def load_model(self):
        model = self.model_class(
            class_num=self.dataset_description["num_classes"],
            features_num=self.features_num,
            with_pca=self.with_pca,
            pca_components=self.pca_components,
        )
        return model

    def process_dataloader(self, train_sampler, val_sampler):
        if self.with_pca:
            train_loader = DataLoader(
                self.dataset,
                batch_size=len(self.dataset),
                sampler=train_sampler,
            )
            val_loader = DataLoader(
                self.dataset,
                batch_size=len(self.dataset),
                sampler=val_sampler,
            )
            train_loader, val_loader = apply_pca(
                train_loader, val_loader, self.batch_size, self.pca_components
            )
        else:
            train_loader, val_loader = super().process_dataloader(train_sampler, val_sampler)
        return train_loader, val_loader
