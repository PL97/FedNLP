import pandas as pd
import os
import torch
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "False"

from models.BERT import BertModel
from datasets.dataset import get_data as get_bert_data
from datasets.BI_LSTM_dataset import get_data as get_bilstm_crf_data
from trainer.trainer_bert import RE_FedAvg_bert
import random
import numpy as np
from collections import defaultdict



import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=str, help="WORKSPACE folder", default="site-1")
    parser.add_argument("--n_split", type=int, help="WORKSPACE folder", default=2)
    parser.add_argument("--model", type=str, help="specify which model to use: [bert-base-uncased/BI_LSTM_CRF]", default="bert-base-uncased")
    parser.add_argument("--batch_size", type=str, help="batchsize of train/val/test loader", default=128)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    args = vars(parse_args())
    saved_dir = args['workspace']
    
    device = torch.device("cuda")


    dls = defaultdict(lambda: {})
    num_client = args['n_split']
    root_dir = f"./data/{num_client}_split"
    
    if 'bert' in args['model'].lower():
        ## need to define tokenizer here
        tokenizer = BertModel(num_labels = 9, model_name=args['model']).tokenizer
        
        for idx in range(num_client):
            dataset_name = f"site-{idx+1}"
            df_train = pd.read_csv(os.path.join(root_dir, dataset_name+"_train.csv"))
            df_val = pd.read_csv(os.path.join(root_dir, dataset_name+"_val.csv"))
            ## for debugging
            dls[idx], stats = get_bert_data(df_train=df_train, df_val=df_val, bs=args['batch_size'], tokenizer=tokenizer)

        fed_model = RE_FedAvg_bert(
                    dls=dls,
                    client_weights = [1/num_client]*num_client, 
                    lrs = [5e-5]*num_client, 
                    max_epoches=50, 
                    aggregation_freq=1,
                    device=device, 
                    saved_dir = saved_dir,
                    model_name=args['model']
                    )
        fed_model.fit()