import torch
from trainer import EvaluatorClassification

from torch.utils.data import DataLoader, SubsetRandomSampler
from data import load_pca_dataset
from sklearn import preprocessing
from sklearn.decomposition import PCA

class EvaluatorTwoBranches(EvaluatorClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model(self):
        model = self.model_class(
            class_num=self.dataset_description["num_classes"],
            with_pca=self.with_pca,
        )
        return model

    def load_model_weights(self, model, fold):
        model_path = self.save_path + "two_branches_" + str(fold) + ".pth"
        model.load_state_dict(torch.load(model_path))

    def process_dataloader(self, train_loader, val_loader):
        if self.with_pca:
            train_loader, val_loader = self.apply_pca(train_loader, val_loader)
        return train_loader, val_loader

    def infer_model(self, model, batch):
        x, x_opensmile, labels = batch
        x, x_opensmile, labels = (
            x.to(self.device),
            x_opensmile.to(self.device),
            labels.to(self.device),
        )
        out = model.forward(x, x_opensmile)
        return labels, out

    def apply_pca(self, train_loader, val_loader):
        x_train, x_opensmile_train, y_train = next(iter(train_loader))
        x_val, x_opensmile_val, y_val = next(iter(val_loader))

        x_opensmile_train = x_opensmile_train.view(-1, 988).numpy()
        x_opensmile_val = x_opensmile_val.view(-1, 988).numpy()

        scaler = preprocessing.StandardScaler()
        x_opensmile_train = scaler.fit_transform(x_opensmile_train)
        x_opensmile_val = scaler.transform(x_opensmile_val)

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