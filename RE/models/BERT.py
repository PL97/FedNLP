from transformers import AutoModelForSequenceClassification, BertTokenizer, BertModel, BertTokenizerFast, AutoTokenizer, AutoModel
import torch
import os

class BertModel(torch.nn.Module):

    def __init__(self, num_labels, model_name='bert-base-uncased', pretrained_path="../../"):

        super(BertModel, self).__init__()

        if model_name == "bluebert":
            self.bert = AutoModelForSequenceClassification.from_pretrained(os.path.join(pretrained_path, "pretrained_models/bluebert/NCBI_BERT_pubmed_uncased_L-12_H-768_A-12/"), \
                        num_labels=num_labels, \
                        output_attentions = False, \
                        output_hidden_states = False)
            self.tokenizer = BertTokenizerFast.from_pretrained(os.path.join(pretrained_path, "pretrained_models/bluebert/NCBI_BERT_pubmed_uncased_L-12_H-768_A-12/"))
        elif model_name == "biobert":
            self.bert = AutoModelForSequenceClassification.from_pretrained("dmis-lab/biobert-v1.1", \
                        num_labels=num_labels, \
                        output_attentions = False, \
                        output_hidden_states = False)
            self.tokenizer = BertTokenizerFast.from_pretrained("dmis-lab/biobert-v1.1")
        elif model_name == "clinicbert":
            self.bert = AutoModelForSequenceClassification.from_pretrained("tdobrxl/ClinicBERT", \
                        num_labels=num_labels, \
                        output_attentions = False, \
                        output_hidden_states = False)
            self.tokenizer = AutoTokenizer.from_pretrained("tdobrxl/ClinicBERT")
        elif model_name == "bert-base-uncased" or model_name == "bert-base-uncased":
            self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, \
                        output_attentions = False, \
                        output_hidden_states = False)
            self.tokenizer = BertTokenizerFast.from_pretrained(model_name)

        else:
            exit("model not found (source: BERT.py)")


    def forward(self, input_ids, attention_mask, labels):

        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        return output