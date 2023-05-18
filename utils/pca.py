from torch.utils.data import DataLoader
from data import load_ser_pca_dataset
from sklearn import preprocessing
from sklearn.decomposition import PCA


def apply_pca(train_loader, val_loader, batch_size, pca_components=100):
    train = next(iter(train_loader))
    val = next(iter(val_loader))

    if len(train) == 3:
        x_train, x_opensmile_train, y_train = train
        x_val, x_opensmile_val, y_val = val
    else:
        x_opensmile_train, y_train = train
        x_opensmile_val, y_val = val

    x_opensmile_train = x_opensmile_train.view(-1, 988).numpy()
    x_opensmile_val = x_opensmile_val.view(-1, 988).numpy()

    scaler = preprocessing.StandardScaler()
    x_opensmile_train = scaler.fit_transform(x_opensmile_train)
    x_opensmile_val = scaler.transform(x_opensmile_val)
    pca = PCA(n_components=pca_components)
    x_train_pca = pca.fit_transform(x_opensmile_train)
    x_val_pca = pca.transform(x_opensmile_val)
    # print(f"explained_variance_ratio_ {pca.explained_variance_ratio_.sum()}")
    if len(train) == 3:
        train_pca = {"x": x_train, "x_opensmile": x_train_pca, "y": y_train}
        val_pca = {"x": x_val, "x_opensmile": x_val_pca, "y": y_val}
        use_keys = ["x", "x_opensmile", "y"]
    else:
        train_pca = {"x_opensmile": x_train_pca, "y": y_train}
        val_pca = {"x_opensmile": x_val_pca, "y": y_val}
        use_keys = ["x_opensmile", "y"]

    train_dataset_pca = load_ser_pca_dataset(train_pca, use_keys)
    val_dataset_pca = load_ser_pca_dataset(val_pca, use_keys)
    train_loader_pca = DataLoader(
        train_dataset_pca,
        batch_size=batch_size,
    )
    val_loader_pca = DataLoader(
        val_dataset_pca,
        batch_size=batch_size,
    )
    return train_loader_pca, val_loader_pca
