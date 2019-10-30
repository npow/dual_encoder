from params import args
import data
from data import load_jsonl, process_train, process_valid

import evaluate
import models
import numpy as np
import torch
from torch import optim
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


L_train = load_jsonl('data/train.jsonl')
L_valid = load_jsonl('data/valid.jsonl')
L_test = load_jsonl('data/test.jsonl')

if use_memory:
    #vocab = data.get_vocab(L_train)
    vocab = data.load_vocab('data/vocabulary.txt')
else:
    vocab = data.load_vocab('data/vocabulary.txt')

L_train = [process_train(vocab, row) for row in tqdm(L_train)]
L_valid = [process_valid(vocab, row) for row in tqdm(L_valid)]
L_test = [process_valid(vocab, row) for row in tqdm(L_test)]


encoder_model = models.Encoder(
    vocab=vocab,
    input_size=input_size,  # embedding dim
    hidden_size=input_size,  # rnn dim
    vocab_size=len(vocab),  # vocab size
    bidirectional=True,  # really should change!
    rnn_type='lstm',
    pretrained_vectors=pretrained_vectors,
)
encoder_model.cuda()

model = models.DualEncoder(encoder_model, use_memory=use_memory)
model.cuda()

loss_fn = torch.nn.BCELoss()
loss_fn.cuda()

num_batches = int(len(L_train)//batch_size)

model_file = '{}/model.pt'.format(checkpoint_dir)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
early_stopping = models.EarlyStopping(model_file=model_file, patience=patience, verbose=True)

for i in range(num_epochs):
    model = model.train()
    losses = []
    recalls = []
    for batch_idx in tqdm(range(num_batches)):
        batch = data.get_batch(L_train, batch_idx, batch_size)

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

    recall_k = evaluate.evaluate(L_valid, model, size=evaluate_batch_size)
    print('epoch: ', i, 'loss: ', np.mean(losses), 'val_recall: ', recall_k)
    early_stopping(recall_k[1], model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

recall_k = evaluate.evaluate(L_test, model, size=evaluate_batch_size)
print('test_recall: ', recall_k)
