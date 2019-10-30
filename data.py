import json


def load_jsonl(fname):
    with open(fname) as f:
        return [json.loads(s.strip()) for s in f.readlines()]


L_train = load_jsonl('data/train.jsonl')[:100]
L_valid = load_jsonl('data/valid.jsonl')[:100]
L_test = load_jsonl('data/test.jsonl')[:100]


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
