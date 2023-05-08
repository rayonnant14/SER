import torch
from trainer import TrainerClassification
from models import TIMNET
import data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default="SAVEE.npy")

def main():
    model = TIMNET()
    args = parser.parse_args()
    dataset_path = args.dataset_path
    train_dataloader, val_dataloader = data.load_dataset(dataset_path)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, betas=(0.93, 0.98))
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = 300
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    trainer = TrainerClassification(model=model,
                                    train_dataloader=train_dataloader,
                                    val_dataloader=val_dataloader,
                                    optimizer=optimizer,
                                    criterion=criterion,
                                    num_epochs=num_epochs,
                                    device=device)
    history = trainer.fit()


if __name__ == '__main__':
    main()
