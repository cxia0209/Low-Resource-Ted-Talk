import torch.nn as nn

class BaseCoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_size, input_dropout, output_dropout, n_layers, rnn):
        super(BaseCoder, self).__init__()
        # init ...
        self.vocab_size = vocab_size
        # self.max_length = max_length
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding_size = embedding_size
        # TODO: why two self.input_dropout here?
        self.input_dropout = input_dropout
        self.input_dropout = nn.Dropout(p=input_dropout)
        self.output_dropout = output_dropout


        if rnn.lower() == "lstm":
            self.baseModel = nn.LSTM
        elif rnn.lower() == "gru":
            self.baseModel = nn.GRU
        else:
            ## raise error
            raise ValueError("No such cell!")
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError