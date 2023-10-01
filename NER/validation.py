import pandas as pd
import os
import torch
from datasets.dataset import get_data as get_bert_data
from datasets.BI_LSTM_dataset import get_data as get_bilstm_crf_data
import argparse
import random
import numpy as np
import json

## custome packages
from models.BERT import BertModel
from models.GPT import GPTModel
from models.BILSTM_CRF import BIRNN_CRF
from datasets.dataset import DataSequence
from trainer.trainer_bert import _shared_validate as validate_bert
from trainer.trainer_bilstm_crf import _shared_validate as validate_bilstm_crf

os.environ["TOKENIZERS_PARALLELISM"] = "False"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, help="dataset name", default="site-1")
    parser.add_argument("--model", type=str, help="specify which model to use: [bert-base-uncased/BI_LSTM_CRF]", default="BI_LSTM_CRF")
    parser.add_argument("--workspace", type=str, help="WORKSPACE folder", default="site-1")
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
    workspace = args['workspace']

    df_train = pd.read_csv(os.path.join(root_dir, args['split']+"_train.csv"))
    df_val = pd.read_csv(os.path.join(root_dir, args['split']+"_val.csv"))
    df_test = pd.read_csv(os.path.join(f"./data/{dataset_name}", "test.csv"))

    
    df_combined = pd.read_csv(os.path.join(f"./data/{dataset_name}", "combined.csv"))
    num_labels = len(set(" ".join(df_combined.labels.tolist()).split(" ")))

    
    ## prepare model
    if "bert" in args['model'].lower():
        ## prepare dataloader
        model = BertModel(num_labels = num_labels, model_name=args['model'])
        ## prepare dataloader
        dls, stats = get_bert_data(df_train=df_train, df_val=df_val, bs=args['batch_size'], tokenizer=model.tokenizer, df_test=df_test, df_combined=df_combined)
    elif "gpt" in args['model'].lower():
        ## prepare model
        model = GPTModel(num_labels = num_labels, model_name=args['model'])
        ## prepare dataloader
        dls, stats = get_bert_data(df_train=df_train, df_val=df_val, bs=args['batch_size'], tokenizer=model.tokenizer, df_test=df_test, df_combined=df_combined)
    
    ## load from pretrained weights
    model.load_state_dict(torch.load(f"{workspace}/best.pt"))
    result_json = {}
    print("training summary")
    result_json['train'] = validate_bert(model, dls['train'], device, stats['ids_to_labels'], "train")
    print("validation summary")
    result_json['validation'] = validate_bert(model, dls['val'], device, stats['ids_to_labels'], "val")
    print("test summary")
    result_json['test'] = validate_bert(model, dls['test'], device, stats['ids_to_labels'], "test")
    
    json_object = json.dumps(result_json, indent=4)
    with open(f"{workspace}/metric_client{args['ds']}.json", "w") as outfile:
        outfile.write(json_object)