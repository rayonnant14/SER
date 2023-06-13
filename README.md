# 🦇 Speech Emotion Recognition 
The repository contains PyTorch implementation of the [TIM-Net](https://arxiv.org/abs/2211.08233) architecture and experiments with combination of TIM-Net and branch which handles additional acoustic and linguistic features. 

The models were tested on 4 datasets:
- SAVEE
- RAVDESS
- EmoDB
- EMOVO

All data is available on Google Drive: https://drive.google.com/drive/folders/1kQ3H61FS0CLNkG0YC5SrLDyC-b4eNgus?usp=sharing


To check all experiments run the following command
```
python3 ./generate_report.py --dataset_path PATH_TO_DATASET_IN_NPY_FORMAT --num_epochs 300 --dataset_name DATASET_NAME --config configs
```

To perform your own experiment modify config file in /data/config_example.py and run
```
python3 ./generate_report.py --dataset_path PATH_TO_DATASET_IN_NPY_FORMAT --num_epochs 300 --dataset_name DATASET_NAME --config config_example
```

