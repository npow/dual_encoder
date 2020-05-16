from params import args
import pickle
import numpy as np
import torch
#torch.manual_seed(42)
#np.random.seed(42)

import data
from data import load_jsonl, process_train, process_valid

import evaluate
import models
from torch import optim
from tqdm import tqdm

def load_data(eval_model):
    vocab = data.load_vocab('data/vocabulary.txt')
    L_test = load_jsonl('data/test.jsonl')
    L_test = [process_valid(vocab, row) for row in tqdm(L_test)]
    if not eval_model:
        L_train = load_jsonl('data/train.jsonl')
        L_valid = load_jsonl('data/valid.jsonl')
        L_train = [process_train(vocab, row) for row in tqdm(L_train)]
        L_valid = [process_valid(vocab, row) for row in tqdm(L_valid)]
    else:
        L_train, L_valid = [], []
    return L_train, L_valid, L_test, vocab

def create_model(
        vocab,
        pretrained_vectors,
        fine_tune_W,
        use_memory,
        hidden_size=300,
        input_size=300,
        rnn_type='lstm',
        bidirectional=False,
        **kwargs,
        ):
    encoder_model = models.Encoder(
        vocab=vocab,
        input_size=input_size,  # embedding dim
        hidden_size=hidden_size,  # rnn dim
        vocab_size=len(vocab),  # vocab size
        bidirectional=bidirectional,
        rnn_type=rnn_type,
        pretrained_vectors=pretrained_vectors,
        fine_tune_W=fine_tune_W,
    )
    encoder_model.cuda()

    knowledge_encoder = models.Encoder(
        vocab=vocab,
        input_size=input_size,  # embedding dim
        hidden_size=hidden_size,  # rnn dim
        vocab_size=len(vocab),  # vocab size
        bidirectional=bidirectional,
        rnn_type=rnn_type,
        pretrained_vectors=pretrained_vectors,
        fine_tune_W=fine_tune_W,
    )
    knowledge_encoder.cuda()

    model = models.DualEncoder(encoder_model, knowledge_encoder, use_memory=use_memory)
    model = model.cuda()

    return model

def do_train(
        model,
        L_train, L_valid, L_test, vocab,
        lr,
        patience,
        batch_size,
        checkpoint_dir,
        num_epochs,
        evaluate_batch_size=None,
        **kwargs,
        ):
    loss_fn = torch.nn.BCELoss()
    loss_fn.cuda()

    num_batches = int(len(L_train)//batch_size)

    model_file = '{}/model.pt'.format(checkpoint_dir)
    optimizer = optim.Adam(model.parameters(), lr=lr)
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
            memory_values = memory_keys
            memory_value_lengths = memory_key_lengths
            #memory_values = torch.LongTensor(batch['memory_values']).cuda()
            #memory_value_lengths = torch.LongTensor(batch['memory_value_lengths']).cuda()

            optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                y_preds, responses = model(contexts=cs, responses=rs,
                                           memory_keys=memory_keys, memory_key_lengths=memory_key_lengths,
                                           memory_values=memory_values, memory_value_lengths=memory_value_lengths)

                loss = loss_fn(y_preds, ys)
                losses.append(loss.item())

                loss.backward()
                optimizer.step()

        recall_k = evaluate.evaluate(L_valid, model, size=evaluate_batch_size)
        print('epoch: ', i, 'loss: ', np.mean(losses), 'val_recall: ', recall_k)
        early_stopping(-recall_k[1], model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    do_eval(model, L_test, evaluate_batch_size=evaluate_batch_size)

def do_eval(model, L_test, evaluate_batch_size=None):
    model = model.eval()
    recall_k = evaluate.evaluate(L_test, model, size=evaluate_batch_size)
    print('test_recall: ', recall_k)

L_train, L_valid, L_test, vocab = load_data(args.eval_only)
if not args.pretrained_model_dir:
    model = create_model(vocab, **args.__dict__)
else:
    with open('{}/params.pkl'.format(args.pretrained_model_dir), 'rb') as f:
        model_params = pickle.load(f)
    model_params['fine_tune_W'] = True
    model_params['batch_size'] = 32
    model = create_model(vocab, **model_params)
    model_dict = model.state_dict()
    pretrained_dict = torch.load('{}/model.pt'.format(args.pretrained_model_dir))
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


if args.force_use_memory:
    model.use_memory = True
    for param in model.parameters():
        param.requires_grad = True
    for param in list(model.bridge.parameters()) + [model.N] + list(model.memory_gate.parameters()):
        param.requires_grad = True

if args.eval_only:
    do_eval(model, L_test)
else:
    do_train(model, L_train, L_valid, L_test, vocab, **args.__dict__)
