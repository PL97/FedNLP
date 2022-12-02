import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
import argparse
import random
import numpy as np
import json

## custome packages
from models.BERT import BertModel
from datasets.dataset import DataSequence
from trainer.trainer import validate

os.environ["TOKENIZERS_PARALLELISM"] = "False"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, help="dataset name", default="site-1")
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
    workspace = args['workspace']
    root_dir = "/home/le/cancerbert_ner/data/"

    df_train = pd.read_csv(os.path.join(root_dir, dataset_name+"_train.csv"))
    df_val = pd.read_csv(os.path.join(root_dir, dataset_name+"_val.csv"))
    df_test = pd.read_csv(os.path.join(root_dir, dataset_name+"_test.csv"))


    print("prepare model")

    model = BertModel(num_labels = 17)
    model.load_state_dict(torch.load(f"{workspace}/best.pt"))
    
    train_dataset = DataSequence(df_train)
    val_dataset = DataSequence(df_val)
    test_dataset = DataSequence(df_test)

    BATCH_SIZE = 8
    device = torch.device("cuda")
    label_map = train_dataset.ids_to_labels
    
    train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)
    val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)

    result_json = {}
    print("training summary")
    result_json['train'] = validate(model, train_dataloader, device, label_map)
    print("validation summary")
    result_json['validation'] = validate(model, val_dataloader, device, label_map)
    print("test summary")
    result_json['test'] = validate(model, test_dataloader, device, label_map)

    json_object = json.dumps(result_json, indent=4)
    with open("metric.json", "w") as outfile:
        outfile.write(json_object)