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
from seqeval.metrics import classification_report, f1_score


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
    
    
def validate(model, dataloader, device, label_map):
    model = model.to(device)
    model.eval()

    total_acc_val = 0
    total_loss_val = 0
    val_total = 0
    val_y_pred = []
    val_y_true = []
    
    for val_data, val_label in dataloader:

        val_label = val_label.to(device)
        val_total += val_label.shape[0]
        mask = val_data['attention_mask'].squeeze(1).to(device)
        input_id = val_data['input_ids'].squeeze(1).to(device)

        loss, logits = model(input_id, mask, val_label)
        

        for i in range(logits.shape[0]):

            logits_clean = logits[i][val_label[i] != -100]
            label_clean = val_label[i][val_label[i] != -100]

            predictions = logits_clean.argmax(dim=1)
            acc = (predictions == label_clean).float().mean()
            
            total_acc_val += acc
            total_loss_val += loss.item()
            val_y_pred.append([label_map[x.item()] for x in predictions])
            val_y_true.append([label_map[x.item()] for x in label_clean])

    print("validation: ", classification_report(val_y_true, val_y_pred))
    print(f1_score(val_y_true, val_y_pred))


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

df_train = pd.read_csv(os.path.join(root_dir, dataset_name+"_train.csv"))
df_val = pd.read_csv(os.path.join(root_dir, dataset_name+"_val.csv"))
df_test = pd.read_csv(os.path.join(root_dir, dataset_name+"_test.csv"))


print("prepare model")

model = BertModel(num_labels = 17)
model.load_state_dict(torch.load(f"{dataset_name}/best.pt"))
# model = DDP(BertModel(num_labels=17), device_ids=[rank], output_device=rank, find_unused_parameters=True)
train_dataset = DataSequence(df_train)
val_dataset = DataSequence(df_val)
test_dataset = DataSequence(df_test)

BATCH_SIZE = 8
device = torch.device("cuda")
label_map = train_dataset.ids_to_labels
# sampler_train = DistributedSampler(train_dataset, num_replicas=4, rank=0, shuffle=False, drop_last=False)
# train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=BATCH_SIZE, sampler=sampler_train)
train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)
val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=False)

print("training summary")
validate(model, train_dataloader, device, label_map)
print("training summary")
validate(model, val_dataloader, device, label_map)
print("training summary")
validate(model, test_dataloader, device, label_map)