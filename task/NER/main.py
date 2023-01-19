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
from trainer.trainer_bilstm_crf import trainer_bilstm_crf
import random
import numpy as np
import argparse




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, help="WORKSPACE folder", default="site-1")
    parser.add_argument("--workspace", type=str, help="WORKSPACE folder", default="site-1")
    parser.add_argument("--model", type=str, help="specify which model to use: [bert-base-uncased/BI_LSTM_CRF]", default="BI_LSTM_CRF")
    parser.add_argument("--batch_size", type=str, help="batchsize of train/val/test loader", default=64)
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



    ## prepare model
    if args['model'] == "bert-base-uncased":
        ## prepare dataloader
        dls, stats = get_bert_data(df_train=df_train, df_val=df_val, bs=args['batch_size'], model_name=args['model'])
        model = BertModel(num_labels = 19, model_name=args['model'])
    
        trainer = trainer_bert(model=model, \
                            dls=dls, \
                            ids_to_labels=stats['ids_to_labels'], \
                            lr=5e-5, \
                            epochs=15, \
                            saved_dir=args['workspace'], \
                            device=device)
        trainer.fit()
    
    elif args['model'] == "BI_LSTM_CRF":
        dls, stats = get_bilstm_crf_data(df_train=df_train, df_val=df_val, bs=args['batch_size'])
        model = BIRNN_CRF(vocab_size=stats['vocab_size'], \
                          tagset_size = len(stats['ids_to_labels'])-2, \
                          embedding_dim=200, \
                          num_rnn_layers=1, \
                          hidden_dim=256, device=device)
        
        trainer = trainer_bilstm_crf(model=model, \
                            dls=dls, \
                            ids_to_labels=stats['ids_to_labels'], \
                            lr=1e-3, \
                            epochs=50, \
                            saved_dir=args['workspace'], \
                            device=device)
        trainer.fit()
        
    

    