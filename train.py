from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

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
num_epochs = 100
batch_size = 512
evaluate_batch_size = None
num_batches = int(10e6//batch_size)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

prev_recall_k = None
for i in range(num_epochs):
    model = model.train()
    losses = []
    recalls = []
    prev_recall_k = None
    for batch_idx in tqdm(range(num_batches)):
        batch = data.get_batch(batch_idx, batch_size)
        batch = list(map(preprocessing.process_train, batch))
        count = 0

        cs, rs, ys = [], [], []
        for c, r, y in batch:
            count += 1

            cs.append(torch.LongTensor(c))
            rs.append(torch.LongTensor(r))
            ys.append(torch.FloatTensor([y]))

        cs = Variable(torch.stack(cs, 0)).cuda()
        rs = Variable(torch.stack(rs, 0)).cuda()
        ys = Variable(torch.stack(ys, 0)).cuda()

        y_preds, responses = model(cs, rs)
        loss = loss_fn(y_preds, ys)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    recall_k = evaluate.evaluate(model, size=evaluate_batch_size, split='dev')
    if prev_recall_k is not None:
        if recall_k[1] > prev_recall_k[1]:
            prev_recall_k = recall_k
            torch.save(model.state_dict(), '{}/model.pt'.format(CHECKPOINT_DIR))
        else:
            print('early stopping!!')
            break
    else:
        prev_recall_k = recall_k
        torch.save(model.state_dict(), '{}/model.pt'.format(CHECKPOINT_DIR))
    print('loss: ', np.mean(losses), 'recall: ', recall_k)

recall_k = evaluate.evaluate(model, size=evaluate_batch_size, split='test')
print('test recall: ', recall_k)
