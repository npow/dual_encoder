import sys
sys.path += ['../qb']

import preprocessing
import torch
import numpy as np

from torch import nn
from torch.nn import init
from torch.nn import Module, Linear, Softmax, CosineSimilarity, Embedding

dtype = torch.cuda.FloatTensor  # Uncomment this to run on GPU


class Encoder(nn.Module):
    def __init__(
            self,
            vocab,
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
        self.vocab = vocab
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

        embeddings = preprocessing.load_embeddings(self.vocab, self.pretrained_vectors)
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
        self.M = nn.Parameter(M, requires_grad=True)

        N = torch.FloatTensor(h_size, h_size).cuda()
        init.normal_(N)
        self.N = nn.Parameter(N, requires_grad=True)

        self.kvmnn = KeyValueMemoryNet(text_embeddings=encoder.embedding, num_classes=None, nn_dropout=nn_dropout)
        self.use_memory = use_memory
        self.memory_gate = nn.Sequential(
            nn.Linear(h_size, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, contexts, responses, memory_keys, memory_key_lengths, memory_values, memory_value_lengths):
        context_lengths = (contexts > 0).sum(dim=-1)

        context_os, _ = self.encoder(contexts)
        context_encodings = context_os[:,-1,:]
        response_os, _ = self.encoder(responses)
        response_encodings = response_os[:,-1,:]
        cMr = torch.bmm((context_encodings @ self.M).unsqueeze(1), response_encodings.unsqueeze(-1)).squeeze(-1)
        results = cMr

        if self.use_memory:
            memory_encodings = self.kvmnn(query=contexts, query_lengths=context_lengths,
                                          memory_keys=memory_keys, memory_key_lengths=memory_key_lengths,
                                          memory_values=memory_values, memory_value_lengths=memory_value_lengths)
            memory_results = (memory_encodings @ self.N)
            #alpha = self.memory_gate(memory_results)
            mNr = torch.bmm(memory_results.unsqueeze(1), response_encodings.unsqueeze(-1)).squeeze(-1)
            results = results + mNr

        results = torch.sigmoid(results)

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
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print('Validation loss decreased ({} --> {}).  Saving model ...'.format(self.val_loss_min, val_loss))
        torch.save(model.state_dict(), self.model_file)
        self.val_loss_min = val_loss


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

class KeyValueMemoryNet(Module):
    """Defines PyTorch model for Key-Value Memory Network.
    Key-Value Memory Networks (KV-MemNN) are described here: https://arxiv.org/pdf/1606.03126.pdf
    Goal is to read correct response from memory, given query. Memory slots are
    defined as pairs (k, v) where k is query and v is correct response. This
    implementation of KV-MemNN uses separate encodings for input query and
    possible candidates. Instead of using cross-entropy loss, we use cosine
    embedding loss where we measure cosine distance between read responses and
    candidate responses. We use only one 'hop' because more hops don't provide
    any improvements.
    This implementation supports batch training.
    """

    def __init__(self, text_embeddings, num_classes, nn_dropout=0., use_memory=True):
        """Initializes model layers.
        Args:
            vocab_size (int): Number of tokens in corpus. This is used to init embeddings.
            embedding_dim (int): Dimension of embedding vector.
        """
        super().__init__()

        vocab_size, embedding_dim = text_embeddings.weight.shape
        self._embedding_dim = embedding_dim

        self.encoder_in = KvmnnEncoder(text_embeddings, nn_dropout=nn_dropout)
        self.encoder_out = KvmnnEncoder(text_embeddings, nn_dropout=nn_dropout)

        if num_classes is None:
            self.linear = Identity()
        else:
            self.linear = nn.Sequential(
                nn.Linear(embedding_dim, num_classes, bias=True),
                nn.BatchNorm1d(num_classes),
                nn.Dropout(nn_dropout)
            )
        self.similarity = CosineSimilarity(dim=2)
        self.softmax = Softmax(dim=2)
        self.use_memory = use_memory

    def forward(self, query, query_lengths, memory_keys, memory_key_lengths, memory_values, memory_value_lengths):
        """Performs forward step.
        Args:
            query (torch.Tensor): Tensor with shape of (NxM) where N is batch size,
               and M is length of padded query.
            memory_keys (torch.Tensor): Relevant memory keys for given query batch. Shape
                of tensor is (NxMxD) where N is batch size, M is number of relevant memories
                per query and D is length of memories.
            memory_values (torch.Tensor): Relevant memory values for given query batch
                with same shape as memory_keys.
            memory_key_lengths: (NxM)
            memory_value_lengths: (NxM)
        """
        view_shape = (len(query), 1, self._embedding_dim)

        query_embedding = self.encoder_in(query, query_lengths).view(*view_shape)

        if self.use_memory:
            memory_keys_embedding = self.encoder_in(memory_keys, memory_key_lengths)
            memory_values_embedding = self.encoder_in(memory_values, memory_value_lengths)

            similarity = self.similarity(query_embedding, memory_keys_embedding).unsqueeze(1)
            softmax = self.softmax(similarity)
            value_reading = torch.matmul(softmax, memory_values_embedding)
            result = self.linear(value_reading.squeeze(1))# + query_embedding.squeeze(1))
        else:
            result = self.linear(query_embedding.squeeze(1))
        return result


class KvmnnEncoder(Module):
    """Embeds queries, memories or responses into vectors."""

    def __init__(self, text_embeddings, nn_dropout=0.):
        """Initializes embedding layer.
        Args:
            num_embeddings (int): Number of possible embeddings.
            embedding_dim (int): Dimension of embedding vector.
        """
        super().__init__()
        self.embedding = text_embeddings #Embedding(*text_embeddings.weight.shape)
        self.dropout = nn.Dropout(nn_dropout)

    def forward(self, tokens, token_lengths):
        tokens_shape = tokens.shape
        if len(tokens_shape) == 3:
            tokens = tokens.view((-1, tokens_shape[-1]))
            token_lengths = token_lengths.view((-1,))
        emb_tokens = self.embedding(tokens)
        token_lengths[token_lengths==0] = 1  # HACK
        retval = emb_tokens.sum(1) / token_lengths.view((tokens.shape[0], -1)).cuda().float()
        if len(tokens_shape) == 3:
            retval = retval.view((tokens_shape[0], tokens_shape[1], self.embedding.embedding_dim))
        return self.dropout(retval)
