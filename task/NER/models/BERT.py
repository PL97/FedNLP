
from transformers import BertForTokenClassification, BertTokenizer, BertModel, BertTokenizerFast
import torch
import os

class BertModel(torch.nn.Module):

    def __init__(self, num_labels, model_name='bert-base-uncased', pretrained_path="../../"):

        super(BertModel, self).__init__()

        if model_name == "bluebert":
            self.bert = BertForTokenClassification.from_pretrained(os.path.join(pretrained_path, "pretrained_models/bluebert/NCBI_BERT_pubmed_uncased_L-12_H-768_A-12/"), \
                        num_labels=num_labels, \
                        output_attentions = False, \
                        output_hidden_states = False)
            self.tokenizer = BertTokenizerFast.from_pretrained(os.path.join(pretrained_path, "pretrained_models/bluebert/NCBI_BERT_pubmed_uncased_L-12_H-768_A-12/"))
        else:
            self.bert = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels, \
                        output_attentions = False, \
                        output_hidden_states = False)
            self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        
        
        
    def forward(self, input_id, mask, label):

        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)

        return output
    

if __name__ == "__main__":
    # pass
    net = BertModel(num_labels=2, model_name="bluebert", pretrained_path="../../../")
    tokenizer = net.tokenizer
    max_length = 150
    input_ids = tokenizer.encode_plus("This is a sample text.", \
                                        padding='max_length', \
                                        max_length = max_length, \
                                        add_special_tokens = True, \
                                        truncation=True, \
                                        return_attention_mask = True)
    print(input_ids)
    outputs = net(torch.tensor(input_ids['input_ids']), torch.tensor(input_ids['attention_mask']), torch.tensor([1, 0]))
    print(outputs)
