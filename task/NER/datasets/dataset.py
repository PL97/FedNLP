import torch
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import re


def align_label(tokenized_inputs, origional_text, labels, labels_to_ids, label_all_tokens=False, tokenizer=None):
    

    word_ids = tokenized_inputs.word_ids()

    # for debug only (remove it after stable release)    
    # print(origional_text.split(" "))
    # print(labels)
    # print(tokenizer.tokenize(origional_text))
    # print(tokenizer.decode(tokenizer.encode(origional_text)))
    # print(len(origional_text.split()), len(labels), len(tokenizer.decode(tokenizer.encode(origional_text)).split()))
    # print(word_ids)
    # print(tokenized_inputs)
    # print(len(labels))

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
            # print("check", sub_str, origional_text[orig_labels_i-1].lower())
            if sub_str == origional_text[orig_labels_i-1].lower():
                partially_mathced = False
                sub_str = ""
            
        # print("parital:{}\tacc_str:{}\torig:{}\t\tcur_str:{}\tids:{}".format(partially_mathced, sub_str, origional_text[orig_labels_i-1], tokenizer.convert_ids_to_tokens(token_id), label_ids[-1]))
    # print(label_ids)
    # print("====="*10)

    return label_ids

class DataSequence(torch.utils.data.Dataset):

    def __init__(self, df, max_length=150, model_name='bert-base-uncased'):
        
        labels = [i.split() for i in df['labels'].values.tolist()]
        unique_labels = set()
        # from collections import Counter
        # print(labels)
        # print(Counter([ll for l in labels for ll in l]))
        
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