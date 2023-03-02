import pandas as pd
import os
import torch
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "False"

from models.BERT import BertModel
from models.BILSTM_CRF import BIRNN_CRF
from models.GPT import GPTModel
from datasets.dataset import get_data as get_bert_data
from datasets.BI_LSTM_dataset import get_data as get_bilstm_crf_data
from trainer.trainer_bert import trainer_bert
from trainer.trainer_gpt import trainer_gpt
from trainer.trainer_bilstm_crf import trainer_bilstm_crf
import random
import numpy as np
import argparse
import json




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, help="WORKSPACE folder", default="2018_n2c2")
    parser.add_argument("--split", type=str, help="WORKSPACE folder", default="site-1")
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
    saved_dir = os.path.join(args['workspace'], args['split'])
    num_client = 10
    root_dir = f"./data/{dataset_name}/{num_client}_split"

    df_train = pd.read_csv(os.path.join(root_dir, args['split']+"_train.csv"))
    df_val = pd.read_csv(os.path.join(root_dir, args['split']+"_val.csv"))
    df_test = pd.read_csv(os.path.join(f"./data/{dataset_name}", "test.csv"))

    
    df_combined = pd.read_csv(os.path.join(f"./data/{dataset_name}", "combined.csv"))
    print(df_combined.labels.tolist())
    num_labels = len(set(df_combined.labels.tolist()))

   
    if "bert" in args['model'].lower():
        ## prepare model
        model = BertModel(num_labels = num_labels, model_name=args['model'])
        ## prepare dataloader
        dls, stats = get_bert_data(df_train=df_train, df_val=df_val, bs=args['batch_size'], tokenizer=model.tokenizer, df_test=df_test, df_combined=df_combined)

        trainer = trainer_bert(model=model, \
                            dls=dls, \
                            ids_to_labels=stats['ids_to_labels'], \
                            lr=5e-5, \
                            epochs=15, \
                            saved_dir=saved_dir, \
                            device=device)
        trainer.fit()
        
    elif "gpt" in args['model'].lower():
        ## prepare model
        model = GPTModel(num_labels = num_labels, model_name=args['model'])
        ## prepare dataloader
        
        dls, stats = get_bert_data(df_train=df_train, df_val=df_val, bs=args['batch_size'], tokenizer=model.tokenizer, df_test=df_test, df_combined=df_combined)
        
        trainer = trainer_gpt(model=model, \
                            dls=dls, \
                            ids_to_labels=stats['ids_to_labels'], \
                            lr=5e-5, \
                            epochs=50, \
                            saved_dir=saved_dir, \
                            device=device)
        trainer.fit()
        
    else:
        exit("cannot find the model (source: main.py)")
    

    metrics = {split: trainer.inference(dls[split], prefix=split) for split in ['train', 'val', 'test']}
    for split in ['train', 'val', 'test']:
        pd.DataFrame(metrics[split]['meta']).to_csv(f"{args['workspace']}/{args['split']}/{split}_prediction.csv")
        metrics[split].pop('meta')        

    with open(f"{args['workspace']}/{args['split']}/evaluation.json", 'w') as f:
        json.dump(metrics, f)