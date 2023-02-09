import torch
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import re
import json


class DataSequence(torch.utils.data.Dataset):

    def __init__(self, df, max_length=150, model_name='bert-base-cased'):
        
        labels = df['relation'].values.tolist()
        unique_labels = set()
        
        for lb in labels:
            [unique_labels.add(i) for i in lb if i not in unique_labels]
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
        self.ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}

        df['text'] = [re.sub(" +", " ", re.sub("[^0-9a-zA-Z%/]", " ", x).strip()) for x in df['text']]
        txt = df['text'].values.tolist()
        self.texts = [tokenizer.encode_plus(i, \
                               padding='max_length', \
                               max_length = max_length, \
                               add_special_tokens = True, \
                               truncation=True, \
                               return_attention_mask = True, \
                               return_tensors="pt") for i in txt]
        
        ## read the data map and convert str into interge
        label_map = dict(json.load(open("/home/le/NLPer/task/RE/data/label_map.json")))
        
        self.labels = list(map(lambda x: label_map[x], labels))
        self.text = txt
        self.ids_to_labels = dict(sorted({v:k for k, v in label_map.items()}.items(), key=lambda item: item[1]))


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
    

def get_data(df_train, df_val, bs, model_name):
    dls, stats = {}, {}
    train_dataset = DataSequence(df_train, model_name=model_name)
    val_dataset = DataSequence(df_val, model_name=model_name)
    dls['train'] = DataLoader(train_dataset, num_workers=4, batch_size=bs, shuffle=True)
    dls['val'] = DataLoader(val_dataset, num_workers=4, batch_size=bs)
    stats['ids_to_labels'] = train_dataset.ids_to_labels
    return dls, stats
    

if __name__ == "__main__":
    import pandas as pd
    import sys
    sys.path.append("../")
    from models.BERT import BertModel
    df = pd.read_csv("../data/1_split/site-1_train.csv")
    train_dataset = DataSequence(df, model_name='bert-base-cased')
    train_dataset = DataLoader(train_dataset, num_workers=4, batch_size=1, shuffle=True)
    model = BertModel(num_labels = 9, model_name='bert-base-cased')
    
    for x, y in train_dataset:
        print(y)
        mask = x['attention_mask'].squeeze(1)
        input_id = x['input_ids'].squeeze(1)
        print(mask.size(), input_id.size(), y.size())
        print(model(input_ids=input_id, attention_mask=mask, labels=y)[:2])
        exit()
    