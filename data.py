import json
import itertools
from collections import Counter
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from params import args
use_memory = args.use_memory != 0

def load_jsonl(fname):
    lines = []
    with open(fname) as f:
        for line in f:
            lines.append(line)
    return [json.loads(s.strip()) for s in tqdm(lines)]


def load_vocab(filename):
    lines = open(filename).readlines()
    return {
        word.strip(): i
        for i, word in enumerate(lines)
    }


def get_vocab(L):
    print('computing vocab')
    cnt = Counter()
    for row in tqdm(L):
        c = row['c']
        r = row['r']
        for token in c.split() + r.split():
            cnt[token] += 1
        for k, v in row['m'].items():
            for s in v + [k]:
                for token in s.split():
                    cnt[token] += 1
    vocab = [word for (word, freq) in cnt.most_common() if freq > 10]
    return {word: i for i, word in enumerate(vocab)}


def get_batch(L_train, epoch, batch_size):
    start = epoch * batch_size % len(L_train)
    return process_train_batch(L_train[start:start + batch_size])


def numberize(vocab, inp):
    inp = inp.split()
    result = list(map(lambda k: vocab.get(k, 0), inp))[-160:]
    return result


def process_sequence(vocab, seq):
    seq = pad_sequences([numberize(vocab, x) for x in seq], padding='post')
    seq_lens = (seq > 0).sum(axis=-1)
    return seq, seq_lens


def process_train(vocab, row):
    cs = numberize(vocab, row['c'])
    rs = numberize(vocab, row['r'])
    y = int(row['y'])

    memory_keys = []
    memory_values = []
    if use_memory:
        for k, v in row['m'].items():
            memory_keys += v
            memory_values += ([k]*len(v))
        if len(memory_keys) < 50:
            memory_keys += ['<pad>'] * (50-len(memory_keys))
            memory_values += ['<pad>'] * (50-len(memory_values))

        memory_keys = [numberize(vocab, x) for x in memory_keys]
        memory_values = [numberize(vocab, x) for x in memory_values]

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

    if use_memory:
        memory_keys = pad_sequences(list(itertools.chain(*[row['memory_keys'] for row in rows]))).reshape((bsz, 50, -1))
        memory_values = pad_sequences(list(itertools.chain(*[row['memory_values'] for row in rows]))).reshape((bsz, 50, -1))
        memory_key_lengths = (memory_keys > 0).sum(-1)
        memory_value_lengths = (memory_values > 0).sum(-1)
    else:
        memory_keys, memory_values, memory_key_lengths, memory_value_lengths = [], [], [], []

    return {
        'cs': cs,
        'rs': rs,
        'ys': ys,
        'memory_keys': memory_keys,
        'memory_key_lengths': memory_key_lengths,
        'memory_values': memory_values,
        'memory_value_lengths': memory_value_lengths,
    }


def process_valid(vocab, row):
    cs = numberize(vocab, row['c'])
    rs = numberize(vocab, row['r'])
    ds = [
        numberize(vocab, distractor)
        for distractor in row['d']
    ]

    memory_keys = []
    memory_values = []
    memory_key_lengths = []
    memory_value_lengths = []
    if use_memory:
        for k, v in row['m'].items():
            memory_keys += v
            memory_values += ([k]*len(v))
        if len(memory_keys) < 50:
            memory_keys += ['<pad>'] * (50-len(memory_keys))
            memory_values += ['<pad>'] * (50-len(memory_values))
        memory_keys, memory_key_lengths = process_sequence(vocab, memory_keys)
        memory_values, memory_value_lengths = process_sequence(vocab, memory_values)

    return {
        'c': cs,
        'r': rs,
        'ds': ds,
        'memory_keys': memory_keys,
        'memory_key_lengths': memory_key_lengths,
        'memory_values': memory_values,
        'memory_value_lengths': memory_value_lengths,
    }
