from transformers import AutoModelForSequenceClassification, BertTokenizer, BertModel, BertTokenizerFast, AutoTokenizer, AutoModel
import torch
import os

class BertModel(torch.nn.Module):

    def __init__(self, num_labels, model_name='bert-base-uncased', pretrained_path="../pretrained_models/"):

        super(BertModel, self).__init__()
        model_name = model_name.lower()
        if model_name == "bluebert":
            self.bert = AutoModelForSequenceClassification.from_pretrained(os.path.join(pretrained_path, "bluebert_pubmed_uncased_L-24_H-1024_A-16/"), \
                        num_labels=num_labels, \
                        output_attentions = False, \
                        output_hidden_states = False)
            self.tokenizer = BertTokenizerFast.from_pretrained(os.path.join(pretrained_path, "bluebert_pubmed_uncased_L-24_H-1024_A-16/"))
        elif model_name == "biobert":
            self.bert = AutoModelForSequenceClassification.from_pretrained(os.path.join(pretrained_path, "biobert-v1.1"), \
                        num_labels=num_labels, \
                        output_attentions = False, \
                        output_hidden_states = False)
            self.tokenizer = BertTokenizerFast.from_pretrained(os.path.join(pretrained_path, "biobert-v1.1"))
        elif model_name == "bio_clinicalbert":
            self.bert = AutoModelForSequenceClassification.from_pretrained(os.path.join(pretrained_path, "Bio_ClinicalBERT"), \
                        num_labels=num_labels, \
                        output_attentions = False, \
                        output_hidden_states = False)
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(pretrained_path, "Bio_ClinicalBERT"))
        elif model_name == "bert-base-uncased":
            self.bert = AutoModelForSequenceClassification.from_pretrained(os.path.join(pretrained_path, "bert-base-uncased"), num_labels=num_labels, \
                        output_attentions = False, \
                        output_hidden_states = False)
            self.tokenizer = BertTokenizerFast.from_pretrained(os.path.join(pretrained_path, "bert-base-uncased"))

        else:
            exit("model not found (source: BERT.py)")


    def forward(self, input_ids, attention_mask, labels):

        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        return output