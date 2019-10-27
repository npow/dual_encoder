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

encoder_model = models.Encoder(
    input_size=100,  # embedding dim
    hidden_size=200,  # rnn dim
    vocab_size=91620,  # vocab size
    bidirectional=True,  # really should change!
    rnn_type='lstm',
)
encoder_model.cuda()

model = models.DualEncoder(encoder_model)
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
    count = 0

    cs, rs, ys = [], [], []
    contexts = []
    for c, r, y in batch:
        count += 1

        cs.append(torch.LongTensor(c))
        rs.append(torch.LongTensor(r))
        ys.append(torch.FloatTensor([y]))
        contexts.append(c)

    cs = Variable(torch.stack(cs, 0)).cuda()
    rs = Variable(torch.stack(rs, 0)).cuda()
    ys = Variable(torch.stack(ys, 0)).cuda()

    y_preds, responses = model(cs, rs, contexts)
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
