from torch import optim
from torch.autograd import Variable

import data
import datetime
import evaluate
import models
import numpy as np
import os
import preprocessing
import time
import torch

CHECKPOINT_DIR = os.environ['CHECKPOINT_DIR']

use_memory = True
encoder_model = models.Encoder(
    input_size=100,  # embedding dim
    hidden_size=200,  # rnn dim
    vocab_size=len(preprocessing.vocab),  # vocab size
    bidirectional=True,  # really should change!
    rnn_type='lstm',
)
encoder_model.cuda()

model = models.DualEncoder(encoder_model, use_memory=use_memory)
model.cuda()

loss_fn = torch.nn.BCELoss()
loss_fn.cuda()

learning_rate = 0.001
num_epochs = 30000
batch_size = 512
evaluate_batch_size = None

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

prev_recall_k = None
for i in range(num_epochs):
    model = model.train()
    batch = data.get_batch(i, batch_size)
    batch = list(map(preprocessing.process_train, batch))

    cs, rs, memory_keys, memory_key_lengths, memory_values, memory_value_lengths, ys = zip(*batch)
    cs = torch.LongTensor(cs).cuda()
    rs = torch.LongTensor(rs).cuda()
    ys = torch.FloatTensor(ys).cuda()
    memory_keys = torch.LongTensor(memory_keys).cuda()
    memory_key_lengths = torch.LongTensor(memory_key_lengths).cuda()
    memory_values = torch.LongTensor(memory_values).cuda()
    memory_value_lengths = torch.LongTensor(memory_value_lengths).cuda()

    y_preds, responses = model(contexts=cs, responses=rs,
                               memory_keys=memory_keys, memory_key_lengths=memory_key_lengths,
                               memory_values=memory_values, memory_value_lengths=memory_value_lengths)
    loss = loss_fn(y_preds, ys)

    recall_k = evaluate.evaluate(model, size=evaluate_batch_size, split='dev')
    if prev_recall_k is not None:
        if recall_k[1] >= prev_recall_k[1]:
            prev_recall_k = recall_k
        else:
            print('early stopping!!')
            break
    else:
        prev_recall_k = recall_k

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    del loss, batch

torch.save(model.state_dict(), '{}/model.pt'.format(CHECKPOINT_DIR))
recall_k = evaluate.evaluate(model, size=evaluate_batch_size, split='test')
print(recall_k)
