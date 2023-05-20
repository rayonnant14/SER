import torch

from trainer import TrainerTwoBranches
from trainer import EvaluatorTwoBranches
from models import TwoBranches, ASRTwoBranches

from data import load_ser_dataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="SAVEE.npy")
parser.add_argument("--dataset_name", type=str, default="SAVEE")
parser.add_argument("--save_path", type=str, default="checkpoints/")
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--label_smoothing", action="store_true")


def main():
    args = parser.parse_args()
    dataset_path = args.dataset_path
    dataset_name = args.dataset_name
    save_path = args.save_path
    num_epochs = args.num_epochs
    label_smoothing = args.label_smoothing
    dataset = load_ser_dataset(dataset_path, use_keys=["x", "x_asr_electra", "y"])

    optimizer_func = torch.optim.Adam
    optimizer_parameters = {"lr": args.lr, "betas": (0.93, 0.98)}
    if label_smoothing:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    with_pca = False
    pca_components = None
    model_class = ASRTwoBranches
    trainer = TrainerTwoBranches(
        dataset=dataset,
        dataset_name=dataset_name,
        model_class=model_class,
        batch_size=batch_size,
        optimizer_func=optimizer_func,
        optimizer_parameters=optimizer_parameters,
        criterion=criterion,
        num_epochs=num_epochs,
        save_path=save_path,
        with_pca=with_pca,
        pca_components=pca_components,
        device=device,
    )
    history = trainer.fit()

    evaluator = EvaluatorTwoBranches(
        dataset=dataset,
        dataset_name=dataset_name,
        model_class=model_class,
        with_pca=with_pca,
        pca_components=pca_components,
        batch_size=batch_size,
        save_path=save_path,
        device=device,
    )
    evaluator.predict()


if __name__ == "__main__":
    main()
