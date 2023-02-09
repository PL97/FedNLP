#encoding:utf-8
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence
from .model_utils import prepare_pack_padded_sequence

class BILSTM(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_layer: int,
                 input_size: int,
                 dropout_p: float,
                 num_classes: int,
                 bi_tag: bool):
        """Bidirectonal LSTM

        Args:
            hidden_size (int): hidden feature dim of lstm
            num_layer (int): number of lstm layers
            input_size (int): input data dimension
            dropout_p (float): dropout rate, no dropout if 0 is specified
            num_classes (int): number of classes to be predicted
            bi_tag (bool): enable bidirectonal
        """

        super(BILSTM,self).__init__()
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

        self.lstm = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = num_layer,
                            batch_first = True,
                            dropout = dropout_p,
                            bidirectional = bi_tag)
        bi_num = 2 if bi_tag else 1
        self.linear = nn.Linear(in_features=hidden_size * bi_num, out_features= num_classes)
        nn.init.xavier_uniform(self.linear.weight)

    def forward(self,inputs,length):
        inputs, length, desorted_indice = prepare_pack_padded_sequence(inputs, length)
        embeddings_packed = pack_padded_sequence(inputs, length, batch_first=True)
        output, _ = self.lstm(embeddings_packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = output[desorted_indice]
        output = F.dropout(output, p=self.dropout_p, training=self.training)
        output = F.tanh(output)
        logit = self.linear(output)
        return logit