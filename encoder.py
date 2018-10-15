import torch
import torch.nn as nn
from generalRnn import BaseCoder

class Encoder(BaseCoder):
    def __init__(self,vocab_size, hidden_size, embedding_size, input_dropout=0.0,output_dropout=0.0, n_layers=1, bidirectional=True,rnn="lstm"):
        super(Encoder, self).__init__(vocab_size, hidden_size, embedding_size, input_dropout,output_dropout, n_layers, rnn)
        self.embedding = nn.Embedding(vocab_size,embedding_size)

        # TODO: add pretrained embeddings

        self.rnn = self.baseModel(input_size=embedding_size, hidden_size=hidden_size, num_layers=n_layers,
                    batch_first=True, bidirectional=bidirectional, dropout=input_dropout)
        for weight in self.rnn.parameters():
            nn.init.uniform_(weight,-0.1, 0.1)

    def forward(self, input_seq, input_lengths=None):
        embedded = self.embedding(input_seq)
        #embedded = self.input_dropout(embedded)
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden
