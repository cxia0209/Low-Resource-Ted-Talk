import torch.nn as nn
import torch

class BaseCoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_size, input_dropout, output_dropout, n_layers, rnn, vocab, embeddings):
        super(BaseCoder, self).__init__()
        # init ...
        self.vocab_size = vocab_size
        # self.max_length = max_length
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # load pre-trained embeddings
        self.load_pretrained_embeddings(vocab, embeddings, trainable=True)

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

    def load_pretrained_embeddings(self, vocab, embeddings, trainable=True):
        if not vocab or not embeddings:
            return
        vocab_size = len(vocab)
        count = 0
        self.embedding.weight.requires_grad = False
        for i in range(vocab_size):
            word = vocab.id2word[i]
            try:
                # embeddings loaded from text
                if 'vectors' not in embeddings or type(embeddings['vectors']) != torch.Tensor:
                    self.embedding.weight[i] = torch.from_numpy(embeddings[word])
                # embeddings loaded from torch bin Dictionary object
                else:
                    self.embedding.weight[i] = embeddings['vectors'][embeddings['dico'].index(word)]
            except Exception as e:
                # print(e)
                count += 1
        print('missing embedding:', count, vocab_size)
        if trainable:
            self.embedding.weight.requires_grad = True

