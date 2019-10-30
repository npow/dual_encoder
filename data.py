import json
from tqdm import tqdm


def load_jsonl(fname):
    with open(fname) as f:
        lines = f.readlines()
    return [json.loads(s.strip()) for s in tqdm(lines)]


L_train = load_jsonl('data/train.jsonl')
L_valid = load_jsonl('data/valid.jsonl')
L_test = load_jsonl('data/test.jsonl')


def get_batch(epoch, batch_size):
    start = epoch * batch_size % len(L_train)
    return L_train[start:start + batch_size]


def get_validation(num=None):
    if num is None:
        return L_valid

    return L_valid[:num]


def get_test(num=None):
    if num is None:
        return L_test

    return L_test[:num]
