import torch

from trainer import TrainerClassification
from trainer import EvaluatorClassification
from models import TIMNET

from data.dataloader import load_dataset
from data.datasets import DATASETS

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="SAVEE.npy")
parser.add_argument("--dataset_name", type=str, default="SAVEE")
parser.add_argument("--save_path", type=str, default="checkpoints/")
parser.add_argument("--num_epochs", type=int, default=2)


def main():
    args = parser.parse_args()
    dataset_path = args.dataset_path
    dataset_name = args.dataset_name
    save_path = args.save_path
    num_epochs = args.num_epochs

    dataset = load_dataset(dataset_path)

    model_parameters = TIMNET(DATASETS[dataset_name]['num_classes']).parameters()
    optimizer = torch.optim.Adam(
        model_parameters, lr=0.001, betas=(0.93, 0.98)
    )
    criterion = torch.nn.CrossEntropyLoss()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = TrainerClassification(
        dataset=dataset,
        dataset_name=dataset_name,
        model_class=TIMNET,
        batch_size=64,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=num_epochs,
        save_path=save_path,
        device=device,
    )
    history = trainer.fit()

    evaluator = EvaluatorClassification(
        dataset=dataset,
        dataset_name=dataset_name,
        model_class=TIMNET,
        batch_size=64,
        save_path=save_path,
        device=device,
    )
    evaluator.evaluate()

if __name__ == "__main__":
    main()
