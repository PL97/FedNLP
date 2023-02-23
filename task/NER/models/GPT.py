
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForCausalLM
import torch

class GPTModel(torch.nn.Module):

    def __init__(self, num_labels, model_name='bert-base-uncased', pretrained_path="../../"):

        super(GPTModel, self).__init__()

        if model_name == "biogpt":
            self.bert = AutoModelForTokenClassification.from_pretrained("microsoft/biogpt", \
                        num_labels=num_labels, \
                        output_attentions = False, \
                        output_hidden_states = False)
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
        elif model_name == "gpt2":
            self.bert = AutoModelForTokenClassification.from_pretrained("gpt2", \
                        num_labels=num_labels, \
                        output_attentions = False, \
                        output_hidden_states = False)
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.bert.resize_token_embeddings(len(self.tokenizer))
        else:
            exit("model not found (source: GPT.py)")
        
        
        
    def forward(self, input_id, mask, label):

        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)

        return output
    

if __name__ == "__main__":
    pass
