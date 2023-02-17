import pandas as pd
import os
import torch
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "False"

from models.BERT import BertModel
from models.BILSTM_CRF import BIRNN_CRF
from datasets.dataset import get_data as get_bert_data
from datasets.BI_LSTM_dataset import get_data as get_bilstm_crf_data
from trainer.trainer_bert import trainer_bert
import random
import numpy as np
import argparse




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, help="WORKSPACE folder", default="site-1")
    parser.add_argument("--workspace", type=str, help="WORKSPACE folder", default="site-1")
    parser.add_argument("--model", type=str, help="specify which model to use: [bert-base-uncased/BI_LSTM_CRF]", default="bert-base-uncased")
    parser.add_argument("--batch_size", type=str, help="batchsize of train/val/test loader", default=128)
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
    root_dir = f"./data/{num_client}_split"

    df_train = pd.read_csv(os.path.join(root_dir, dataset_name+"_train.csv"))
    df_val = pd.read_csv(os.path.join(root_dir, dataset_name+"_val.csv"))



   
    if "bert" in args['model'].lower():
        ## prepare model
        model = BertModel(num_labels = 9, model_name=args['model'])
        ## prepare dataloader
        dls, stats = get_bert_data(df_train=df_train, df_val=df_val, bs=args['batch_size'], tokenizer=model.tokenizer)
        
        trainer = trainer_bert(model=model, \
                            dls=dls, \
                            ids_to_labels=stats['ids_to_labels'], \
                            lr=5e-5, \
                            epochs=15, \
                            saved_dir=saved_dir, \
                            device=device)
        trainer.fit()