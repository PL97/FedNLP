import pandas as pd


from tqdm import tqdm
import os
import torch
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from torch.optim import SGD
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from BERT import BertModel
from dataset import align_label
from dataset import DataSequence
from trainer import train_loop


df = pd.read_csv("ner.csv")


  
        
import numpy as np
print(df.shape)
df = df[0:100]
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                            [int(.8 * len(df)), int(.9 * len(df))])

labels = [i.split() for i in df['labels'].values.tolist()]
unique_labels = set()

for lb in labels:
  [unique_labels.add(i) for i in lb if i not in unique_labels]

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}

model = BertModel(num_labels = len(unique_labels))
train_dataset = DataSequence(df_train, tokenizer, labels_to_ids)
val_dataset = DataSequence(df_val, tokenizer, labels_to_ids)
train_args = {
    "LEARNING_RATE": 5e-3,
    "EPOCHS": 5,
    "BATCH_SIZE": 2
}
train_loop(model, train_dataset, val_dataset, **train_args)