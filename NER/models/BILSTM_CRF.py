import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.CRF import CRF


class BIRNN_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim, num_rnn_layers=1, rnn="lstm", device=None):
        super(BIRNN_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.device = device

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        RNN = nn.LSTM if rnn == "lstm" else nn.GRU
        self.rnn = RNN(embedding_dim, hidden_dim // 2, num_layers=num_rnn_layers,
                       bidirectional=True, batch_first=True)
        self.crf = CRF(hidden_dim, self.tagset_size)
        self.dropout = nn.Dropout(0.5)
        
        

    def __build_features(self, sentences):
        masks = sentences.gt(0)
        embeds = self.embedding(sentences.long())
        
        seq_length = masks.sum(1)
        sorted_seq_length, perm_idx = seq_length.sort(descending=True)
        embeds = embeds[perm_idx, :]
        
        pack_sequence = pack_padded_sequence(embeds, lengths=sorted_seq_length.cpu(), batch_first=True)
        packed_output, _ = self.rnn(pack_sequence)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        _, unperm_idx = perm_idx.sort()
        lstm_out = lstm_out[unperm_idx, :]
        lstm_out = self.dropout(lstm_out)

        return lstm_out, masks

    def loss(self, xs, tags):
        features, masks = self.__build_features(xs)
        loss = self.crf.loss(features, tags, masks=masks)
        return loss

    def forward(self, xs):
        # Get the emission scores from the BiLSTM
        features, masks = self.__build_features(xs)
        scores, tag_seq = self.crf(features, masks)
        return scores, tag_seq
    
if __name__ == "__main__":
    pass
    # net = BIRNN_CRF(vocab_size=100000, \
    #                       tagset_size = 10, \
    #                       embedding_dim=200, \
    #                       num_rnn_layers=1, \
    #                       hidden_dim=256, device=None)

    # import sys
    # sys.path.append("../")
    # from utils.utils import count_parameters
    
    # count_parameters(net)
