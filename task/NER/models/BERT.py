
from transformers import BertForTokenClassification
import torch

class BertModel(torch.nn.Module):

    def __init__(self, num_labels):

        super(BertModel, self).__init__()

        # self.bert = BertForTokenClassification.from_pretrained('bert-large-cased', num_labels=num_labels)
        self.bert = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=num_labels)

    def forward(self, input_id, mask, label):

        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)

        return output