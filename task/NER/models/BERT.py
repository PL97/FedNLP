
from transformers import BertForTokenClassification
import torch

class BertModel(torch.nn.Module):

    def __init__(self, num_labels, model_name='bert-base-uncased'):

        super(BertModel, self).__init__()

        self.bert = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels, \
                    output_attentions = False, \
                    output_hidden_states = False)
        
    def forward(self, input_id, mask, label):

        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)

        return output