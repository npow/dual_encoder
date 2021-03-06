import csv
import random

reader = csv.reader(open('data/train.csv'))
rows = list(reader)[1:]
random.shuffle(rows)


def get_batch(epoch, batch_size):
    start = epoch * batch_size % len(rows)
    return rows[start:start + batch_size]


reader = csv.reader(open('data/valid.csv'))
valid = list(reader)[1:]

reader = csv.reader(open('data/test.csv'))
test = list(reader)[1:]


def get_validation(num=None):
    if num is None:
        return valid

    return valid[:num]
    # return [random.choice(valid) for _ in range(num)]


def get_test(num=None):
    if num is None:
        return test

    return test[:num]
    # return [random.choice(test) for _ in range(num)]
