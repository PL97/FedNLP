import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import defaultdict


class DataSequence(torch.utils.data.Dataset):
    
    def __init__(self, df, labels_to_ids, ids_to_labels, word_to_ids, max_length=20):
        self.labels_to_ids, self.ids_to_labels, self.word_to_ids = labels_to_ids, ids_to_labels, word_to_ids
        self.max_length = max_length
        self.labels_orig = [i.split() for i in df['labels'].values.tolist()]
        self.texts_orig = [i.split() for i in df['text'].values.tolist()]

        ## get the index of the label and words in text
        self.labels = [list(map(lambda x: self.labels_to_ids[x], l)) for l in self.labels_orig]
        self.texts = [list(map(lambda x: self.word_to_ids[x], t)) for t in self.texts_orig]
        
        ## padding and truncate to ensure same length
        self.label_pad, self.text_pad = [], []
        for l, t in zip(self.labels, self.texts):
             ## padding or truncate data
            t_pad = [self.word_to_ids['<PAD>']]*self.max_length
            t_pad[:min(self.max_length, len(t))] = t[:min(self.max_length, len(t))]
            l_pad = [self.labels_to_ids['<PAD>']]*self.max_length
            l_pad[:min(self.max_length, len(l))] = l[:min(self.max_length, len(l))]
            self.text_pad.append(t_pad)
            self.label_pad.append(l_pad)
        self.texts = self.text_pad
        self.labels = self.label_pad
        
        
    def __len__(self):
        return len(self.labels)

    def get_batch_data(self, idx):
        return torch.LongTensor(self.texts[idx])

    def get_batch_labels(self, idx):
        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)
        return batch_data, batch_labels


def preprocessing(train_df):
    labels_orig = [i.split() for i in train_df['labels'].values.tolist()]
    texts_orig = [i.split() for i in train_df['text'].values.tolist()]
    unique_labels = set()
    unique_words = set()
    for lb, txt in zip(labels_orig, texts_orig):
        [unique_labels.add(i) for i in lb if i not in unique_labels]
        [unique_words.add(j) for j in txt if j not in unique_words]
    labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
    ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
    word_to_ids = defaultdict(lambda: 0)
    ## unkown words are indexed as 0
    word_to_ids.update({k:v+1 for v, k in enumerate(sorted(unique_words))})
    
    
    ## add start and stop, pad tokens in the label maps, add unknown words to word map
    labels_to_ids.update({
        "<PAD>": len(unique_labels), \
        "<START>": len(unique_labels)+1, \
        "<STOP>": len(unique_labels)+2
    })
    
    ids_to_labels.update({
        len(unique_labels): "<PAD>", \
        len(unique_labels)+1: "<START>",  \
        len(unique_labels)+2: "<STOP>", 
    })
    
    word_to_ids.update({
        '<PAD>': len(word_to_ids)+1, \
        '<UNK>': 0
    })
    
    unique_words.add('<PAD>')
    unique_words.add('<UNK>')
    return labels_to_ids, ids_to_labels, word_to_ids, unique_labels


def get_data(df_train, df_val, bs, combined_df):
    
    dls, stats = {}, {}
    # df = pd.read_csv("./data/2018_Track_2_ADE_and_medication_extraction_challenge/ner.csv")
    labels_to_ids, ids_to_labels, word_to_ids, unique_labels = preprocessing(train_df=combined_df)
    dls['train'] = torch.utils.data.DataLoader(
                        DataSequence(df_train, max_length=75, \
                        labels_to_ids=labels_to_ids, ids_to_labels=ids_to_labels, \
                        word_to_ids=word_to_ids), \
                        batch_size=bs, shuffle=True, num_workers=4)
    dls['val'] = torch.utils.data.DataLoader(
                        DataSequence(df_val, max_length=75, \
                        labels_to_ids=labels_to_ids, ids_to_labels=ids_to_labels, \
                        word_to_ids=word_to_ids), \
                        batch_size=bs, shuffle=False, num_workers=4)
    stats['vocab_size'] = len(word_to_ids)
    stats['ids_to_labels'] = ids_to_labels
    return dls, stats

if __name__ == "__main__":
    pass
    # import pandas as pd
    # df = pd.read_csv("../data/2018_Track_2_ADE_and_medication_extraction_challenge/2_split/site-1_train.csv")
    # labels_to_ids, ids_to_labels, word_to_ids, unique_labels = preprocessing(train_df=df)
    # train_loader = torch.utils.data.DataLoader(
    #                         DataSequence(df, max_length=10, \
    #                         labels_to_ids=labels_to_ids, ids_to_labels=ids_to_labels, \
    #                         word_to_ids=word_to_ids), \
    #                     batch_size=10, shuffle=True, num_workers=8)
    # for x, y in train_loader:
    #     print(x.shape)