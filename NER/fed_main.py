import pandas as pd
import os
import torch
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "False"

from models.BERT import BertModel
from models.GPT import GPTModel
from datasets.dataset import get_data as get_bert_data
from datasets.BI_LSTM_dataset import get_data as get_bilstm_crf_data
from trainer.trainer_bert import NER_FedAvg_bert, NER_FedProx_bert
from trainer.trainer_bilstm_crf import NER_FedAvg_bilstm_crf, NER_FedProx_bilstm_crf
from trainer.trainer_gpt import NER_FedAvg_gpt, NER_FedProx_gpt
import random
import numpy as np
from collections import defaultdict
import json



import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, help="WORKSPACE folder", default="2018_n2c2")
    parser.add_argument("--workspace", type=str, help="WORKSPACE folder", default="site-1")
    parser.add_argument("--n_split", type=int, help="WORKSPACE folder", default=2)
    parser.add_argument("--model", type=str, help="specify which model to use: [bert-base-uncased/BI_LSTM_CRF]", default="bert-base-uncased")
    parser.add_argument("--batch_size", type=str, help="batchsize of train/val/test loader", default=64)
    parser.add_argument("--epochs", type=int, help="total training epochs", default=1)
    parser.add_argument("--eval", action='store_true', help="evaluate best model")
    parser.add_argument("--seed", type=int, help="radom seed", default=0)
    parser.add_argument("--fedalg", type=str, help="federated learning algorithm", default="fedavg")
    parser.add_argument("--mu", type=float, help="mu (fedprox)", default=0.1)
    parser.add_argument("--train_from_scratch", action='store_true', help="train the model from scratch")
    parser.add_argument("--test_file", type=str, help="batchsize of train/val/test loader", default="test")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = vars(parse_args())

    seed = args['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    saved_dir = args['workspace']
    
    device = torch.device("cuda")


    dls = defaultdict(lambda: {})
    num_client = args['n_split']
    root_dir = f"./data/{args['ds']}/{num_client}_split"

    df_combined = pd.read_csv(os.path.join(f"./data/{args['ds']}", "combined.csv"))
    num_labels = len(set(" ".join(df_combined.labels.tolist()).split(" ")))
    print(args['test_file'])
    df_test = pd.read_csv(os.path.join(f"./data/{args['ds']}", f"{args['test_file']}.csv"))
    if "bert" in args['model'].lower():
        ## need to define tokenizer here
        tokenizer = BertModel(num_labels = num_labels, model_name=args['model']).tokenizer
        
        for idx in range(num_client):    
            dataset_name = f"site-{idx+1}"
            df_train = pd.read_csv(os.path.join(root_dir, dataset_name+"_train.csv"))
            df_val = pd.read_csv(os.path.join(root_dir, dataset_name+"_val.csv"))
            dls[idx], stats = get_bert_data(df_train=df_train, df_val=df_val, bs=args['batch_size'], tokenizer=tokenizer, df_test=df_test, df_combined=df_combined)
            
        ## prepare models
        if args['fedalg'].lower() == "fedavg":
            fed_model = NER_FedAvg_bert(dls=dls, \
                                        client_weights = [1/num_client]*num_client,  \
                                        lrs = [5e-5]*num_client,  \
                                        max_epoches=args['epochs'],  \
                                        aggregation_freq=1, \
                                        device=device,  \
                                        saved_dir = saved_dir, \
                                        model_name=args['model'], \
                                        num_labels=num_labels, \
                                        ids_to_labels=stats['ids_to_labels'],  \
                                        amp=True)
        elif args['fedalg'].lower() == "fedprox":
            fed_model = NER_FedProx_bert(dls=dls, \
                                        client_weights = [1/num_client]*num_client,  \
                                        lrs = [5e-5]*num_client,  \
                                        max_epoches=args['epochs'],  \
                                        aggregation_freq=50, \
                                        device=device,  \
                                        saved_dir = saved_dir, \
                                        model_name=args['model'], \
                                        num_labels=num_labels, \
                                        ids_to_labels=stats['ids_to_labels'],  \
                                        amp=True, \
                                        mu=args['mu'])
        else:
            exit("federated learning algorithm not found (source: fed_main.py)")
        if not args['eval']:      
            fed_model.fit()
    
    elif "gpt" in args['model'].lower():
        ## need to define tokenizer here
        tokenizer = GPTModel(num_labels = num_labels, model_name=args['model']).tokenizer
        
        for idx in range(num_client):    
            dataset_name = f"site-{idx+1}"
            df_train = pd.read_csv(os.path.join(root_dir, dataset_name+"_train.csv"))
            df_val = pd.read_csv(os.path.join(root_dir, dataset_name+"_val.csv"))
            dls[idx], stats = get_bert_data(df_train=df_train, df_val=df_val, bs=args['batch_size'], tokenizer=tokenizer, df_test=df_test, df_combined=df_combined)
            
        ## prepare models
        if args['fedalg'].lower() == "fedavg":
            fed_model = NER_FedAvg_gpt(dls=dls, \
                                    client_weights = [1/num_client]*num_client,  \
                                    lrs = [5e-5]*num_client,  \
                                    max_epoches=args['epochs'],  \
                                    aggregation_freq=1, \
                                    device=device,  \
                                    saved_dir = saved_dir, \
                                    model_name=args['model'], \
                                    num_labels=num_labels, \
                                    ids_to_labels=stats['ids_to_labels'], \
                                    amp=True)
        elif args['fedalg'].lower() == "fedprox":  
            saved_dir = os.path.join(saved_dir, str(args['mu']))
            fed_model = NER_FedProx_gpt(dls=dls, \
                                    client_weights = [1/num_client]*num_client,  \
                                    lrs = [5e-5]*num_client,  \
                                    max_epoches=args['epochs'],  \
                                    aggregation_freq=1, \
                                    device=device,  \
                                    saved_dir = saved_dir, \
                                    model_name=args['model'], \
                                    num_labels=num_labels, \
                                    ids_to_labels=stats['ids_to_labels'], \
                                    amp=True,\
                                    mu=args['mu']) 
        else:
            exit("federated learning algorithm not found (source: fed_main.py)")
        if not args['eval']:
            fed_model.fit()
        
    elif args['model'].lower() == "bi_lstm_crf":
        for idx in range(num_client):
            dataset_name = f"site-{idx+1}"
            df_train = pd.read_csv(os.path.join(root_dir, dataset_name+"_train.csv"))
            df_val = pd.read_csv(os.path.join(root_dir, dataset_name+"_val.csv"))
            ## for debugging
            dls[idx], stats = get_bilstm_crf_data(df_train=df_train, df_val=df_val, bs=args['batch_size'], combined_df=df_combined, df_test=df_test)

        
        if args['fedalg'].lower() == "fedavg":
            fed_model = NER_FedAvg_bilstm_crf(
                        dls=dls,
                        client_weights = [1/num_client]*num_client, 
                        lrs = [1e-3]*num_client, 
                        max_epoches=args['epochs'], 
                        aggregation_freq=1,
                        device=device, 
                        saved_dir = saved_dir,
                        model_name=args['model'],
                        vocab_size=stats['vocab_size'], 
                        ids_to_labels=stats['ids_to_labels'],
                        num_labels=num_labels, \
                        amp=True)
        elif args['fedalg'].lower() == "fedprox": 
            saved_dir = os.path.join(saved_dir, str(args['mu']))
            fed_model = NER_FedProx_bilstm_crf(
                        dls=dls,
                        client_weights = [1/num_client]*num_client, 
                        lrs = [1e-3]*num_client, 
                        max_epoches=args['epochs'], 
                        aggregation_freq=1,
                        device=device, 
                        saved_dir = saved_dir,
                        model_name=args['model'],
                        vocab_size=stats['vocab_size'], 
                        ids_to_labels=stats['ids_to_labels'],
                        num_labels=num_labels, \
                        amp=True, \
                        mu=args['mu'])
        else:
            exit("federated learning algorithm not found (source: fed_main.py)")
        if not args['eval']: 
            fed_model.fit()
    
    else:
        exit("cannot find the model (source: main.py)")
    
    fed_model.server_model.load_state_dict(torch.load(f"./{args['workspace']}/global/best.pt"))
    metrics = {split: fed_model.inference(dls[0][split], split) for split in ['test']}
    print(metrics)
    for split in ['test']:
        pd.DataFrame(metrics[split]['meta']).to_csv(f"{args['workspace']}/{split}_prediction.csv")
        metrics[split].pop('meta')
    
    with open(f"{args['workspace']}/evaluation.json", 'w') as f:
        json.dump(metrics, f)
    
    
     