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

import sys
sys.path.append(".")
from model.BILSTM_CRF import BIRNN_CRF



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, help="WORKSPACE folder", default="site-1")
    parser.add_argument("--workspace", type=str, help="WORKSPACE folder", default="site-1")
    parser.add_argument("--model", type=str, help="specify which model to use: [BERT/BI_LSTM_CRF]", default="Bert")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda")
    args = vars(parse_args())
    dataset_name = args['ds']
    saved_dir = os.path.join(args['workspace'], args['ds'])
    num_client = 10
    root_dir = f"./data/2018_Track_2_ADE_and_medication_extraction_challenge/{num_client}_split"

    df_train = pd.read_csv(os.path.join(root_dir, dataset_name+"_train.csv"))
    df_val = pd.read_csv(os.path.join(root_dir, dataset_name+"_val.csv"))

    ## prepare dataloader
    train_dataset = DataSequence(df_train, model_name=model_name)
    val_dataset = DataSequence(df_val, model_name=model_name)
    train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=32)


    ## prepare model
    if args['model'] == "BERT":
        model_name='bert-base-uncased'
        model = BertModel(num_labels = 19, model_name=model_name)
        
    
    elif args['model'] == "BI_LSTM_CRF":
        model = BIRNN_CRF(vocab_size=stats['vocab_size'], \
                          tagset_size = len(args.ids_to_labels)-2, \
                          embedding_dim=200, \
                          num_rnn_layers=1, \
                          hidden_dim=256, device=device)
    

    train_args = {
        "LEARNING_RATE": 5e-5,
        "EPOCHS": 15,
        "device": device
    }
    train_loop(model, train_dataloader, val_dataloader, saved_dir=saved_dir, **train_args)