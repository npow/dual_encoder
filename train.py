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
    hidden_size=100,  # rnn dim
    vocab_size=len(preprocessing.vocab),  # vocab size
    bidirectional=True,  # really should change!
    rnn_type='lstm',
    pretrained_vectors='glove',
)
encoder_model.cuda()

model = models.DualEncoder(encoder_model, use_memory=use_memory)
model.cuda()

loss_fn = torch.nn.BCELoss()
loss_fn.cuda()

learning_rate = 0.001
num_epochs = 100
batch_size = 32
evaluate_batch_size = None
num_batches = int(10e6//batch_size)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

prev_recall_k = None
for i in range(num_epochs):
    model = model.train()
    losses = []
    recalls = []
    for batch_idx in tqdm(range(num_batches)):
        batch = data.get_batch(batch_idx, batch_size)
        batch = preprocessing.process_train_batch(batch)

        cs = torch.LongTensor(batch['cs']).cuda()
        rs = torch.LongTensor(batch['rs']).cuda()
        ys = torch.FloatTensor(batch['ys']).cuda()
        memory_keys = torch.LongTensor(batch['memory_keys']).cuda()
        memory_key_lengths = torch.LongTensor(batch['memory_key_lengths']).cuda()
        memory_values = torch.LongTensor(batch['memory_values']).cuda()
        memory_value_lengths = torch.LongTensor(batch['memory_value_lengths']).cuda()

        with torch.autograd.set_detect_anomaly(True):
            y_preds, responses = model(contexts=cs, responses=rs,
                                       memory_keys=memory_keys, memory_key_lengths=memory_key_lengths,
                                       memory_values=memory_values, memory_value_lengths=memory_value_lengths)

            loss = loss_fn(y_preds, ys)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    recall_k = evaluate.evaluate(model, size=evaluate_batch_size, split='dev')
    if prev_recall_k is not None:
        if recall_k[1] >= prev_recall_k[1]:
            prev_recall_k = recall_k
            torch.save(model.state_dict(), '{}/model.pt'.format(CHECKPOINT_DIR))
        else:
            print('early stopping!!')
            break
    else:
        prev_recall_k = recall_k
        torch.save(model.state_dict(), '{}/model.pt'.format(CHECKPOINT_DIR))
    print('epoch: ', i, 'loss: ', np.mean(losses), 'recall: ', recall_k)

recall_k = evaluate.evaluate(model, size=evaluate_batch_size, split='test')
print('test recall: ', recall_k)
