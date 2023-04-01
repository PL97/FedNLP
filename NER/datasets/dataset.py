import torch
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import re


def align_label(tokenized_inputs, labels, labels_to_ids):
    null_label_id = -100

    # generate label id vector for the network
    # mark the tokens to be ignored
    labels_aligned = []
    # single sentence each time, so always use 0 index
    # get the index mapping from token to word
    # this can be dependent on the specific tokenizer
    word_ids = tokenized_inputs.word_ids(batch_index=0)
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            # set None the ignore tokens
            labels_aligned.append(null_label_id)
        elif word_idx != previous_word_idx:
            # only label the first token of a word
            labels_aligned.append(labels_to_ids[labels[word_idx]])
        else:
            labels_aligned.append(null_label_id)
        previous_word_idx = word_idx
    return labels_aligned

class DataSequence(torch.utils.data.Dataset):

    def __init__(self, df, labels_to_ids, ids_to_labels, tokenizer, max_length=150):

        # Raw texts and corresponding labels
        texts_batch_raw = [i.split(" ") for i in df["text"].values.tolist()]
        labels_batch_raw = [i.split(" ") for i in df["labels"].values.tolist()]
        # Iterate through all cases
        self.texts = []
        self.labels = []
        for batch_idx in range(len(texts_batch_raw)):
            texts_raw = texts_batch_raw[batch_idx]
            labels_raw = labels_batch_raw[batch_idx]
            # Encode texts with tokenizer
            texts_encoded = tokenizer.encode_plus(
                texts_raw,
                padding="max_length",
                max_length=max_length,
                add_special_tokens=True,
                truncation=True,
                is_split_into_words=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            labels_aligned = align_label(texts_encoded, labels_raw, labels_to_ids)
            self.texts.append(texts_encoded)
            self.labels.append(labels_aligned)


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
    
def preprocess(df_combined):
    labels = []
    for x in df_combined['labels'].values:
        labels.extend(x.split(" "))
    unique_labels = set(labels)
    
    labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
    ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
    return labels_to_ids, ids_to_labels

def get_data(df_train, df_val, bs, tokenizer, df_test=None, df_combined=None):
    dls, stats = {}, {}
    labels_to_ids, ids_to_labels = preprocess(df_combined) if df_combined is not None else preprocess(df_combined)
    train_dataset = DataSequence(df_train, labels_to_ids, ids_to_labels, tokenizer=tokenizer)
    val_dataset = DataSequence(df_val, labels_to_ids, ids_to_labels, tokenizer=tokenizer)
    dls['train'] = DataLoader(train_dataset, num_workers=4, batch_size=bs, shuffle=True)
    dls['val'] = DataLoader(val_dataset, num_workers=4, batch_size=bs, shuffle=False)
    if df_test is not None:
        test_dataset = DataSequence(df_test, labels_to_ids, ids_to_labels, tokenizer=tokenizer)
        dls['test'] = DataLoader(test_dataset, num_workers=4, batch_size=bs, shuffle=False)
    stats['ids_to_labels'] = ids_to_labels
    return dls, stats
    