import torch
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def align_label(texts, labels, tokenizer, labels_to_ids, label_all_tokens=True, max_length=150):
    
    
    
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=max_length, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids

class DataSequence(torch.utils.data.Dataset):

    def __init__(self, df, max_length=150):
        
        labels = [i.split() for i in df['labels'].values.tolist()]
        unique_labels = set()
        # from collections import Counter
        # print(labels)
        # print(Counter([ll for l in labels for ll in l]))
        
        for lb in labels:
            [unique_labels.add(i) for i in lb if i not in unique_labels]
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
        self.ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}

        lb = [i.split() for i in df['labels'].values.tolist()]
        txt = df['text'].values.tolist()
        self.texts = [tokenizer(str(i),
                               padding='max_length', max_length = max_length, truncation=True, return_tensors="pt") for i in txt]
        self.labels = [align_label(i,j, labels_to_ids=labels_to_ids, tokenizer=tokenizer, max_length=max_length) for i,j in zip(txt, lb)]

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