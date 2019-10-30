import sys
sys.path += ['../qb']

import preprocessing
import torch
import numpy as np

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
            pretrained_vectors='glove',
    ):
        super(Encoder, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size // self.num_directions
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.pretrained_vectors = pretrained_vectors

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
        init.orthogonal_(self.rnn.weight_ih_l0)
        init.uniform_(self.rnn.weight_hh_l0, a=-0.01, b=0.01)

        embeddings = preprocessing.load_embeddings(self.pretrained_vectors)
        embedding_weights = torch.FloatTensor(self.vocab_size, self.input_size)
        init.uniform_(embedding_weights, a=-0.25, b=0.25)
        for k, v in embeddings.items():
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
        init.normal_(M)
        self.M = nn.Parameter(
            M,
            requires_grad=True,
        )
        self.kvmnn = KeyValueMemoryNet(text_embeddings=encoder.embedding, num_classes=None, nn_dropout=nn_dropout)
        self.use_memory = use_memory

    def forward(self, contexts, responses, memory_keys, memory_key_lengths, memory_values, memory_value_lengths):
        context_lengths = (contexts > 0).sum(dim=-1)

        context_os, _ = self.encoder(contexts)
        context_encodings = context_os[:,-1,:]
        response_os, _ = self.encoder(responses)
        response_encodings = response_os[:,-1,:]

        if self.use_memory:
            memory_encodings = self.kvmnn(query=contexts, query_lengths=context_lengths,
                                           memory_keys=memory_keys, memory_key_lengths=memory_key_lengths,
                                           memory_values=memory_values, memory_value_lengths=memory_value_lengths)
            context_encodings = context_encodings + memory_encodings

        results = torch.bmm((context_encodings @ self.M).unsqueeze(1), response_encodings.unsqueeze(-1)).squeeze()
        results = torch.sigmoid(results).unsqueeze(1)

        return results, response_encodings


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, model_file, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model_file = model_file

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.model_file)
        self.val_loss_min = val_loss