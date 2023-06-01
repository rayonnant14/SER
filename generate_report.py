import torch
import pandas as pd

from data import load_ser_dataset
from data import DATASETS

import argparse
from IPython.display import display
from tqdm import tqdm
import importlib 
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="SAVEE.npy")
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--save_path", type=str, default="checkpoints/")
parser.add_argument(
    "--report_drop_path",
    type=str,
    default="/Users/polina/Documents/diploma_2023/report.csv",
)
parser.add_argument("--num_epochs", type=int, default=300)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--label_smoothing", action="store_true")


def main():
    args = parser.parse_args()
    dataset_path = args.dataset_path
    dataset_name = args.dataset_name
    save_path = args.save_path
    num_epochs = args.num_epochs
    label_smoothing = args.label_smoothing
    os.environ["SEED"] = str(DATASETS[dataset_name]["seed"])
    optimizer_func = torch.optim.Adam
    optimizer_parameters = {"lr": args.lr, "betas": (0.93, 0.98)}
    if label_smoothing:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    module = importlib.import_module('data.' + args.config)
    configs = module.configs
    result_dict = {
        "model_class": [],
        "features": [],
        "with_pca": [],
        "fusion": [],
        "WAR": [],
        "UAR": [],
    }
    base_arguments = {
        "dataset_name": dataset_name,
        "batch_size": batch_size,
        "optimizer_func": optimizer_func,
        "optimizer_parameters": optimizer_parameters,
        "criterion": criterion,
        "num_epochs": num_epochs,
        "save_path": save_path,
        "device": device,
    }
    with tqdm(total=len(configs)) as pbar:
        for config in configs:
            for with_pca in [False]:
                dataset = load_ser_dataset(
                    dataset_path, use_keys=config["use_keys"]
                )
                additional_model_parameters = {
                    "dataset": dataset,
                    "with_pca": with_pca,
                }
                input_parameters = {
                    **base_arguments,
                    **config["trainer_inputs"],
                    **additional_model_parameters,
                }
                trainer = config["trainer"](**input_parameters)
                history = trainer.fit()
                metrics = trainer.predict()

                result_dict["model_class"].append(
                    config["trainer_inputs"]["model_class"].__name__
                )
                result_dict["features"].append(config["use_keys"])
                result_dict["with_pca"].append(with_pca)
                if "fusion" in config["trainer_inputs"]:
                    result_dict["fusion"].append(
                        config["trainer_inputs"]["fusion"].__name__
                    )
                else:
                    result_dict["fusion"].append('None')
                result_dict["WAR"].append(metrics["average_WAR"])
                result_dict["UAR"].append(metrics["average_UAR"])
            pbar.update(1)
    df = pd.DataFrame(data=result_dict)
    df = df.sort_values(by='UAR', ascending=False)
    display(df)
    df.to_csv(args.report_drop_path)


if __name__ == "__main__":
    main()
