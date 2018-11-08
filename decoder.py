import torch
import torch.nn as nn
import torch.nn.functional as F
from generalRnn import BaseCoder
from attention import Attention

import numpy as np


class Decoder(BaseCoder):
    def __init__(self, vocab_size, hidden_size, embedding_size, input_dropout=0.0, output_dropout=0.0, n_layers=1,
                 bidirectional=False, rnn="lstm", tf_rate=0.9, vocab=None, embeddings=None):
        super(Decoder,self).__init__(vocab_size, hidden_size,embedding_size,input_dropout,output_dropout, n_layers, rnn, vocab, embeddings)
        self.rnn = self.baseModel(input_size=embedding_size, hidden_size=hidden_size, num_layers=n_layers, 
                    batch_first=True,dropout=output_dropout)
        self.output_size = vocab_size

        # temporary set attention embedding size to hidden size
        self.attention = Attention(self.hidden_size)
        self.wsm = nn.Linear(self.hidden_size, self.output_size)
        self.attention_mlp = nn.Linear(2*embedding_size, embedding_size)
        self.tf_rate = tf_rate


    def forward(self, input_seq, encoder_hidden, encoder_outputs, func=F.log_softmax, stage="train"):

        # batch_size = input_seq.size(0)
        # TODO: max_length should be fixed for test decoding?
        max_length = input_seq.size(1)
        # using cuda or not
        inputs = input_seq
        
        # for bidrectional encoder
        # encoder_hidden: (num_layers * num_directions, batch_size, hidden_size)
        decoder_hidden = tuple([torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2) for h in encoder_hidden])

        outputs = []
        symbols = None      
        
        prev = inputs[:, 0].unsqueeze(1)
        attention_hidden = None

        for i in range(max_length):
            softmax, decoder_hidden, attention,attention_hidden = self.forward_helper(prev, decoder_hidden,encoder_outputs,attention_hidden, func)
            output_seq = softmax.squeeze(1) # batch * seq_length
            outputs.append(output_seq)

            if i == max_length - 1: break
            if np.random.random_sample() < self.tf_rate:
                prev = inputs[:, i+1].unsqueeze(1)
            else:
                prev = outputs[-1].topk(1)[1] # max probability index
            if stage != "train":
                prev = outputs[-1].topk(1)[1]
            
            if symbols is None: 
                symbols = prev
            else:
                symbols = torch.cat([symbols,prev],dim=1)
        return outputs,decoder_hidden,symbols



    # could insert one parameter like: src_matrix
    def forward_helper(self, decoder_input, decoder_hidden, encoder_outputs,attention_prev, func):
        batch_size = decoder_input.size(0)
        output_size = decoder_input.size(1)
        embedded = self.embedding(decoder_input)
        if attention_prev is not None:
            att = self.attention_mlp(attention_prev.view(-1, 2*embedded.size(2))).view(batch_size, -1, embedded.size(2))
            embeded = torch.mul(embedded, att)
        output,hidden = self.rnn(embedded, decoder_hidden)
        output, attention = self.attention(output, encoder_outputs) # Attention        
        softmax = func(self.wsm(output.view(-1, self.hidden_size)), dim=1).view(batch_size,output_size,-1)
        return softmax, hidden, attention, output
