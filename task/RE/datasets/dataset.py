import torch
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import re
import json


class DataSequence(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer, labels_to_ids, ids_to_labels, max_length=150):
        
        labels = df['labels'].values.tolist()
        
        # df['text'] = [re.sub(" +", " ", re.sub("[^0-9a-zA-Z%/]", " ", x).strip()) for x in df['text']]
        df['text'] = [re.sub(" +", " ", x.strip()) for x in df['text']]
        txt = df['text'].values.tolist()
        self.texts = [tokenizer.encode_plus(i, \
                               padding='max_length', \
                               max_length = max_length, \
                               add_special_tokens = True, \
                               truncation=True, \
                               return_attention_mask = True, \
                               return_tensors="pt") for i in txt]
        
        # ## read the data map and convert str into interge
        # label_map = dict(json.load(open("./data/label_map.json")))
        
        self.labels = list(map(lambda x: labels_to_ids[x], labels))
        self.text = txt


    def __len__(self):

        return len(self.labels)

    def get_batch_data(self, idx):

        return self.texts[idx]

    def get_batch_labels(self, idx):
        return self.labels[idx]
        # return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):

        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)
        return batch_data, batch_labels
    
def preprocess(df_combined):
    labels = df_combined['labels'].values.tolist()
    unique_labels = set(labels)
    
    labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
    ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
    return labels_to_ids, ids_to_labels

def get_data(df_train, df_val, bs, tokenizer, df_test=None, df_combined=None):
    dls, stats = {}, {}
    labels_to_ids, ids_to_labels = preprocess(df_combined) if df_combined is not None else preprocess(df_combined)
    train_dataset = DataSequence(df_train, tokenizer, labels_to_ids, ids_to_labels)
    val_dataset = DataSequence(df_val, tokenizer, labels_to_ids, ids_to_labels)
    dls['train'] = DataLoader(train_dataset, num_workers=4, batch_size=bs, shuffle=True)
    dls['val'] = DataLoader(val_dataset, num_workers=4, batch_size=bs)
    if df_test is not None:
        test_dataset = DataSequence(df_test, tokenizer, labels_to_ids, ids_to_labels)
        dls['test'] = DataLoader(test_dataset, num_workers=4, batch_size=bs)
    stats['ids_to_labels'] = ids_to_labels
    return dls, stats
    

if __name__ == "__main__":
    pass
    # import pandas as pd
    # import sys
    # sys.path.append("../")
    # from models.BERT import BertModel
    # df = pd.read_csv("../data/1_split/site-1_train.csv")
    # train_dataset = DataSequence(df, model_name='bert-base-cased')
    # train_dataset = DataLoader(train_dataset, num_workers=4, batch_size=1, shuffle=True)
    # model = BertModel(num_labels = 9, model_name='bert-base-cased')
    
    # for x, y in train_dataset:
    #     print(y)
    #     mask = x['attention_mask'].squeeze(1)
    #     input_id = x['input_ids'].squeeze(1)
    #     print(mask.size(), input_id.size(), y.size())
    #     print(model(input_ids=input_id, attention_mask=mask, labels=y)[:2])
    #     exit()
    