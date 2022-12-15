import pandas as pd
import os
import torch
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "False"

from models.BERT import BertModel
from datasets.dataset import DataSequence
from trainer.trainer import train_loop
import random
import numpy as np


import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, help="WORKSPACE folder", default="site-1")
    parser.add_argument("--workspace", type=str, help="WORKSPACE folder", default="site-1")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    args = vars(parse_args())
    dataset_name = args['ds']
    saved_dir = os.path.join(args['workspace'], args['ds'])
    root_dir = "./data/2018_Track_2_ADE_and_medication_extraction_challenge/"

    df_train = pd.read_csv(os.path.join(root_dir, dataset_name+"_train.csv"))
    df_val = pd.read_csv(os.path.join(root_dir, dataset_name+"_val.csv"))


    print("prepare model")

    model = BertModel(num_labels = 19)
    train_dataset = DataSequence(df_train)
    val_dataset = DataSequence(df_val)
    train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=32)

    device = torch.device("cuda")

    print("get model ready")

    train_args = {
        "LEARNING_RATE": 5e-5,
        "EPOCHS": 50,
        "device": device
    }
    train_loop(model, train_dataloader, val_dataloader, saved_dir=saved_dir, **train_args)