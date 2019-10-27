import sys
sys.path += ['../qb']

import preprocessing
import torch

from torch import nn
from torch.nn import init
from qanta.util.kvmnn import KeyValueMemoryNet

dtype = torch.cuda.FloatTensor  # Uncomment this to run on GPU


class Encoder(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            vocab_size,
            num_layers=1,
            dropout=0,
            bidirectional=True,
            rnn_type='gru',
    ):
        super(Encoder, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size // self.num_directions
        self.num_layers = num_layers
        self.rnn_type = rnn_type

        self.embedding = nn.Embedding(vocab_size, input_size, sparse=False, padding_idx=0)

        if rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size,
                self.hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                batch_first=True,
            ).cuda()
        else:
            self.rnn = nn.LSTM(
                input_size,
                self.hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                batch_first=True,
            ).cuda()

        self.init_weights()

    def forward(self, inps):
        embs = self.embedding(inps)
        outputs, hiddens = self.rnn(embs)
        return outputs, hiddens

    def init_weights(self):
        init.orthogonal(self.rnn.weight_ih_l0)
        init.uniform(self.rnn.weight_hh_l0, a=-0.01, b=0.01)

        glove_embeddings = preprocessing.load_glove_embeddings()
        embedding_weights = torch.FloatTensor(self.vocab_size, self.input_size)
        init.uniform(embedding_weights, a=-0.25, b=0.25)
        for k, v in glove_embeddings.items():
            embedding_weights[k] = torch.FloatTensor(v)
        embedding_weights[0] = torch.FloatTensor([0] * self.input_size)
        del self.embedding.weight
        self.embedding.weight = nn.Parameter(embedding_weights)


class DualEncoder(nn.Module):
    def __init__(self, encoder, use_memory=False, nn_dropout=0.0):
        super(DualEncoder, self).__init__()
        self.encoder = encoder
        h_size = self.encoder.hidden_size * self.encoder.num_directions
        M = torch.FloatTensor(h_size, h_size).cuda()
        init.normal(M)
        self.M = nn.Parameter(
            M,
            requires_grad=True,
        )
        dense_dim = 2 * self.encoder.hidden_size
        self.dense = nn.Linear(dense_dim, dense_dim).cuda()
        if use_memory:
            self.kvmnn = KeyValueMemoryNet(text_embeddings=encoder.embedding, num_classes=None, nn_dropout=nn_dropout)

    def forward(self, contexts, responses):
        context_os, _ = self.encoder(contexts)
        response_os, _ = self.encoder(responses)

        response_encodings = response_os[:,-1,:]
        results = torch.bmm((context_os[:,-1,:] @ self.M).unsqueeze(1), response_encodings.unsqueeze(-1)).squeeze()
        results = torch.sigmoid(results)

        return results, response_encodings
