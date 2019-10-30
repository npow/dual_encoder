import json
import itertools
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_jsonl(fname):
    with open(fname) as f:
        lines = f.readlines()
    return [json.loads(s.strip()) for s in tqdm(lines)]


def load_vocab(filename):
    lines = open(filename).readlines()
    return {
        word.strip(): i
        for i, word in enumerate(lines)
    }


vocab = load_vocab('data/vocabulary.txt')


def get_batch(epoch, batch_size):
    start = epoch * batch_size % len(L_train)
    return process_train_batch(L_train[start:start + batch_size])


def get_validation(num=None):
    if num is None:
        return L_valid

    return L_valid[:num]


def get_test(num=None):
    if num is None:
        return L_test

    return L_test[:num]


def numberize(inp):
    inp = inp.split()
    result = list(map(lambda k: vocab.get(k, 0), inp))[-160:]
    return result


def process_sequence(seq):
    seq = pad_sequences([numberize(x) for x in seq], padding='post')
    seq_lens = (seq > 0).sum(axis=-1)
    return seq, seq_lens


def process_train(row):
    cs = numberize(row['c'])
    rs = numberize(row['r'])
    y = int(row['y'])

    memory_keys = []
    memory_values = []
    for k, v in row['m'].items():
        memory_keys += v
        memory_values += ([k]*len(v))
    if len(memory_keys) < 50:
        memory_keys += ['<pad>'] * (50-len(memory_keys))
        memory_values += ['<pad>'] * (50-len(memory_values))

    memory_keys = [numberize(x) for x in memory_keys]
    memory_values = [numberize(x) for x in memory_values]

    return {
        'c': cs,
        'r': rs,
        'y': y,
        'memory_keys': memory_keys,
        'memory_values': memory_values,
    }


def process_train_batch(rows):
    bsz = len(rows)
    cs = pad_sequences([row['c'] for row in rows], maxlen=160)
    rs = pad_sequences([row['r'] for row in rows], maxlen=160)
    ys = [row['y'] for row in rows]

    memory_keys = pad_sequences(list(itertools.chain(*[row['memory_keys'] for row in rows]))).reshape((bsz, 50, -1))
    memory_values = pad_sequences(list(itertools.chain(*[row['memory_values'] for row in rows]))).reshape((bsz, 50, -1))
    memory_key_lengths = (memory_keys > 0).sum(-1)
    memory_value_lengths = (memory_values > 0).sum(-1)

    return {
        'cs': cs,
        'rs': rs,
        'ys': ys,
        'memory_keys': memory_keys,
        'memory_key_lengths': memory_key_lengths,
        'memory_values': memory_values,
        'memory_value_lengths': memory_value_lengths,
    }


def process_valid(row):
    cs = numberize(row['c'])
    rs = numberize(row['r'])
    ds = [
        numberize(distractor)
        for distractor in row['d']
    ]

    memory_keys = []
    memory_values = []
    for k, v in row['m'].items():
        memory_keys += v
        memory_values += ([k]*len(v))
    if len(memory_keys) < 50:
        memory_keys += ['<pad>'] * (50-len(memory_keys))
        memory_values += ['<pad>'] * (50-len(memory_values))
    memory_keys, memory_key_lengths = process_sequence(memory_keys)
    memory_values, memory_value_lengths = process_sequence(memory_values)

    return {
        'c': cs,
        'r': rs,
        'ds': ds,
        'memory_keys': memory_keys,
        'memory_key_lengths': memory_key_lengths,
        'memory_values': memory_values,
        'memory_value_lengths': memory_value_lengths,
    }


L_train = [process_train(row) for row in tqdm(load_jsonl('data/train.jsonl'))]
L_valid = [process_valid(row) for row in tqdm(load_jsonl('data/valid.jsonl'))]
L_test = [process_valid(row) for row in tqdm(load_jsonl('data/test.jsonl'))]


