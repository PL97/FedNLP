import pandas as pd


from tqdm import tqdm
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from torch.nn import DataParallel
os.environ["TOKENIZERS_PARALLELISM"] = "False"
import torch.distributed as dist

from BERT import BertModel
from dataset import align_label
from dataset import DataSequence
from trainer import train_loop
import random
import numpy as np


import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, help="WORKSPACE folder", default="site-1")
    args = parser.parse_args()
    return args

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# rank = 4
# setup(rank=rank, world_size=4)
args = vars(parse_args())
dataset_name = args['ds']
root_dir = "/home/le/cancerbert_ner/data/"

## get data statistics
# df = pd.read_csv(os.path.join(root_dir, "ner.csv"))
# train_dataset = DataSequence(df)

df_train = pd.read_csv(os.path.join(root_dir, dataset_name+"_train.csv")).sample(1000)
df_val = pd.read_csv(os.path.join(root_dir, dataset_name+"_val.csv")).sample(50)


print("prepare model")

model = BertModel(num_labels = 17)
# model = DDP(BertModel(num_labels=17), device_ids=[rank], output_device=rank, find_unused_parameters=True)
train_dataset = DataSequence(df_train)
val_dataset = DataSequence(df_val)

print("get model ready")

train_args = {
    "LEARNING_RATE": 5e-3,
    "EPOCHS": 10,
    "BATCH_SIZE": 16
}
train_loop(model, train_dataset, val_dataset, saved_dir=dataset_name, **train_args)
# cleanup()