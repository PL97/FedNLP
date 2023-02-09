
from transformers import BertForSequenceClassification
import torch

class BertModel(torch.nn.Module):

    def __init__(self, num_labels, model_name='bert-base-uncased'):

        super(BertModel, self).__init__()

        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, \
                    output_attentions = False, \
                    output_hidden_states = False)
        
        
    def forward(self, input_ids, attention_mask, labels):

        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        return output