import argparse
import pickle

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

argparser = argparse.ArgumentParser()
argparser.add_argument('--checkpoint_dir', type=str, default='.')
argparser.add_argument('--pretrained_vectors', type=str, default='glove')
argparser.add_argument('--lr', type=float, default=0.001)
argparser.add_argument('--num_epochs', type=int, default=100)
argparser.add_argument('--batch_size', type=int, default=32)
argparser.add_argument('--patience', type=int, default=10)
argparser.add_argument('--use_memory', type=int, default=0)
args = argparser.parse_args()
with open('{}/params.pkl'.format(args.checkpoint_dir), 'wb') as f:
    pickle.dump(args.__dict__, f)

import data
import datetime
import evaluate
import models
import numpy as np
import os
import preprocessing
import time
import torch
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

pretrained_vectors = args.pretrained_vectors
checkpoint_dir = args.checkpoint_dir
learning_rate = args.lr
num_epochs = args.num_epochs
batch_size = args.batch_size
evaluate_batch_size = None
patience = args.patience
input_size = 100 if pretrained_vectors == 'glove' else 300
use_memory = args.use_memory != 0

encoder_model = models.Encoder(
    input_size=input_size,  # embedding dim
    hidden_size=input_size,  # rnn dim
    vocab_size=len(data.vocab),  # vocab size
    bidirectional=True,  # really should change!
    rnn_type='lstm',
    pretrained_vectors=pretrained_vectors,
)
encoder_model.cuda()

model = models.DualEncoder(encoder_model, use_memory=use_memory)
model.cuda()

loss_fn = torch.nn.BCELoss()
loss_fn.cuda()

num_batches = int(len(data.L_train)//batch_size)

model_file = '{}/model.pt'.format(checkpoint_dir)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
early_stopping = models.EarlyStopping(model_file=model_file, patience=patience, verbose=True)

for i in range(num_epochs):
    model = model.train()
    losses = []
    recalls = []
    for batch_idx in tqdm(range(num_batches)):
        batch = data.get_batch(batch_idx, batch_size)

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
    print('epoch: ', i, 'loss: ', np.mean(losses), 'val_recall: ', recall_k)
    early_stopping(recall_k[1], model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

recall_k = evaluate.evaluate(model, size=evaluate_batch_size, split='test')
print('test_recall: ', recall_k)
