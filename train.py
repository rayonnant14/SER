import torch
from trainer import TrainerClassification
from models import TIMNET
import data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="SAVEE.npy")
parser.add_argument("--dataset_name", type=str, default="SAVEE")
parser.add_argument("--save_path", type=str, default="checkpoints/")


def main():
    model = TIMNET(class_num=7)
    args = parser.parse_args()
    dataset_path = args.dataset_path
    dataset_name = args.dataset_name
    save_path = args.save_path
    dataset = data.load_dataset(dataset_path)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, betas=(0.93, 0.98)
    )
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = TrainerClassification(
        dataset=dataset,
        dataset_name=dataset_name,
        batch_size=64,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=num_epochs,
        save_path=save_path,
        device=device,
    )
    history = trainer.fit()


if __name__ == "__main__":
    main()
