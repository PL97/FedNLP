
from transformers import BertForSequenceClassification, BertTokenizer, BertModel, BertTokenizerFast
import torch
import os

class BertModel(torch.nn.Module):

    def __init__(self, num_labels, model_name='bert-base-uncased', pretrained_path="../../"):

        super(BertModel, self).__init__()

        if model_name == "bluebert":
            self.bert = BertForSequenceClassification.from_pretrained(os.path.join(pretrained_path, "pretrained_models/bluebert/NCBI_BERT_pubmed_uncased_L-12_H-768_A-12/"), \
                        num_labels=num_labels, \
                        output_attentions = False, \
                        output_hidden_states = False)
            self.tokenizer = BertTokenizerFast.from_pretrained(os.path.join(pretrained_path, "pretrained_models/bluebert/NCBI_BERT_pubmed_uncased_L-12_H-768_A-12/"))
        else:
            self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, \
                        output_attentions = False, \
                        output_hidden_states = False)
            self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        
        
    def forward(self, input_ids, attention_mask, labels):

        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        return output