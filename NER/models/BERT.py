
from transformers import AutoModelForTokenClassification, BertTokenizer, BertModel, BertTokenizerFast, AutoTokenizer, AutoModel
import torch
import os

class BertModel(torch.nn.Module):

    def __init__(self, num_labels, model_name='bert-base-uncased', pretrained_path="../pretrained_models/"):

        super(BertModel, self).__init__()

        model_name = model_name.lower()
        if model_name == "bluebert":
            self.bert = AutoModelForTokenClassification.from_pretrained(os.path.join(pretrained_path, "bluebert_pubmed_uncased_L-24_H-1024_A-16/"), \
                        num_labels=num_labels, \
                        output_attentions = False, \
                        output_hidden_states = False)
            self.tokenizer = BertTokenizerFast.from_pretrained(os.path.join(pretrained_path, "bluebert_pubmed_uncased_L-24_H-1024_A-16/"))
        elif model_name == "biobert":
            self.bert = AutoModelForTokenClassification.from_pretrained(os.path.join(pretrained_path, "biobert-v1.1"), \
                        num_labels=num_labels, \
                        output_attentions = False, \
                        output_hidden_states = False)
            self.tokenizer = BertTokenizerFast.from_pretrained(os.path.join(pretrained_path, "biobert-v1.1"))
        elif model_name == "bio_clinicalbert":
            self.bert = AutoModelForTokenClassification.from_pretrained(os.path.join(pretrained_path, "Bio_ClinicalBERT"), \
                        num_labels=num_labels, \
                        output_attentions = False, \
                        output_hidden_states = False)
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(pretrained_path, "Bio_ClinicalBERT"))
        elif model_name == "bert-base-uncased":
            self.bert = AutoModelForTokenClassification.from_pretrained(os.path.join(pretrained_path, "bert-base-uncased"), num_labels=num_labels, \
                        output_attentions = False, \
                        output_hidden_states = False)
            self.tokenizer = BertTokenizerFast.from_pretrained(os.path.join(pretrained_path, "bert-base-uncased"))
        else:
            exit("model not found (source: BERT.py)")
        
        
    def forward(self, input_id, mask, label):
        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
        return output
    

if __name__ == "__main__":
    pass
    # net = BertModel(num_labels=2, model_name="bert-base-uncased", pretrained_path="../../pretrained_models/")
    
    # import sys
    # sys.path.append("../")
    # from utils.utils import count_parameters
    
    # count_parameters(net)
    
    # tokenizer = net.tokenizer
    # max_length = 150
    # input_ids = tokenizer.encode_plus("This is a sample text.", \
    #                                     padding='max_length', \
    #                                     max_length = max_length, \
    #                                     add_special_tokens = True, \
    #                                     truncation=True, \
    #                                     return_attention_mask = True)
    # print(input_ids)
    # outputs = net(torch.tensor(input_ids['input_ids']), torch.tensor(input_ids['attention_mask']), torch.tensor([1, 0]))
    # print(outputs)
