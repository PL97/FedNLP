import pandas as pd
import os
import torch
from datasets.dataset import get_data as get_bert_data
import argparse
import random
import numpy as np
import json

## custome packages
from models.BERT import BertModel
from models.BILSTM_CRF import BIRNN_CRF
from datasets.dataset import DataSequence
from trainer.trainer_bert import _shared_validate as validate_bert

os.environ["TOKENIZERS_PARALLELISM"] = "False"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, help="dataset name", default="site-1")
    parser.add_argument("--model", type=str, help="specify which model to use: [bert-base-uncased/BI_LSTM_CRF]", default="bert-base-uncased")
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
    workspace = args['workspace']
    num_client = 1
    root_dir = f"./data/{num_client}_split"

    df_train = pd.read_csv(os.path.join(root_dir, dataset_name+"_train.csv"))
    df_val = pd.read_csv(os.path.join(root_dir, dataset_name+"_val.csv"))
    df_test = pd.read_csv("./data/test.corp.csv")

    
    ## prepare model
    if "bert" in args['model']:
        ## prepare dataloader
        dls, stats = get_bert_data(df_train=df_train, df_val=df_val, df_test=df_test, bs=args['batch_size'], model_name=args['model'])
        model = BertModel(num_labels = 9, model_name=args['model'])
        model.load_state_dict(torch.load(f"{workspace}/final.pt"))
        model = model.to(device)
        
        result_json = {}
        print("training summary")
        result_json['train'] = validate_bert(model, dls['train'], device, stats['ids_to_labels'], "train")
        print("validation summary")
        result_json['validation'] = validate_bert(model, dls['val'], device, stats['ids_to_labels'], "val")
        print("test summary")
        try:
            result_json['test'] = validate_bert(model, dls['test'], device, stats['ids_to_labels'], "test")
        except:
            print("test file has no label")
        
        json_object = json.dumps(result_json, indent=4)
        with open(f"{workspace}/metric_client{args['ds']}.json", "w") as outfile:
            outfile.write(json_object)