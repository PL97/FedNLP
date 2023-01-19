import torch
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import re


def align_label(tokenized_inputs, origional_text, labels, labels_to_ids, label_all_tokens=False, tokenizer=None):
    null_label_id = -100
    label_ids = []
    origional_text = origional_text.split(" ")


    orig_labels_i = 0
    partially_mathced = False
    sub_str = str()
    for token_id in tokenized_inputs['input_ids'][0]:
        token_id = token_id.numpy().item()
        cur_str = tokenizer.convert_ids_to_tokens(token_id).lower()
        if (token_id == tokenizer.pad_token_id) or \
            (token_id == tokenizer.cls_token_id) or \
            (token_id == tokenizer.sep_token_id):
            
            label_ids.append(null_label_id)
            
        elif (not partially_mathced) and \
            origional_text[orig_labels_i].lower().startswith(cur_str) and \
            origional_text[orig_labels_i].lower() != cur_str:
            
            label_str = labels[orig_labels_i]
            label_ids.append(labels_to_ids[label_str])
            orig_labels_i += 1
            partially_mathced = True
            sub_str += cur_str
        
        elif (not partially_mathced) and origional_text[orig_labels_i].lower() == cur_str:
            label_str = labels[orig_labels_i]
            label_ids.append(labels_to_ids[label_str])
            orig_labels_i += 1
            partially_mathced = False

        else:
            label_ids.append(null_label_id)
            sub_str += re.sub("#+", "", cur_str)
            if sub_str == origional_text[orig_labels_i-1].lower():
                partially_mathced = False
                sub_str = ""

    return label_ids

class DataSequence(torch.utils.data.Dataset):

    def __init__(self, df, max_length=150, model_name='bert-base-uncased'):
        
        labels = [i.split() for i in df['labels'].values.tolist()]
        unique_labels = set()
        
        for lb in labels:
            [unique_labels.add(i) for i in lb if i not in unique_labels]
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
        self.ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}

        lb = [i.split(" ") for i in df['labels'].values.tolist()]
        txt = df['text'].values.tolist()
        self.texts = [tokenizer.encode_plus(str(i), \
                               padding='max_length', \
                               max_length = max_length, \
                               add_special_tokens = True, \
                               truncation=True, \
                               return_attention_mask = True, \
                               return_tensors="pt") for i in txt]
        self.labels = [align_label(t, tt, l, labels_to_ids=labels_to_ids, tokenizer=tokenizer) \
                        for t, tt, l in zip(self.texts, txt, lb)]


    def __len__(self):

        return len(self.labels)

    def get_batch_data(self, idx):

        return self.texts[idx]

    def get_batch_labels(self, idx):

        return torch.LongTensor(self.labels[idx])

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
    