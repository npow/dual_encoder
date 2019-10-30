import nltk
from nltk.stem import SnowballStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_vocab(filename):
    lines = open(filename).readlines()
    return {
        word.strip(): i
        for i, word in enumerate(lines)
    }


vocab = load_vocab('data/vocabulary.txt')


def load_glove_embeddings(filename='data/glove.6B.100d.txt'):
    lines = open(filename).readlines()
    embeddings = {}
    for line in lines:
        word = line.split()[0]
        embedding = list(map(float, line.split()[1:]))
        if word in vocab:
            embeddings[vocab[word]] = embedding

    return embeddings


def numberize(inp):
    inp = inp.split()
    result = list(map(lambda k: vocab.get(k, 0), inp))[-160:]
    if len(result) < 160:
        result = [0] * (160 - len(result)) + result
    return result


def process_sequence(seq):
    seq = pad_sequences([numberize(x) for x in seq], padding='post')
    seq_lens = (seq > 0).sum(dim=-1)
    return seq, seq_lens


def process_train(row):
    context, response, memory_keys, memory_values, label = row

    context = numberize(context)
    response = numberize(response)
    label = int(label)
    memory_keys, memory_key_lengths = process_sequence(memory_keys)
    memory_values, memory_value_lengths = process_sequence(memory_values)

    return context, response, memory_keys, memory_key_lengths, memory_values, memory_value_lengths, label


def process_valid(row):
    context = row[0]
    response = row[1]
    memory_keys = row[2]
    memory_values = row[3]
    distractors = row[4:]

    context = numberize(context)
    response = numberize(response)
    memory_keys, memory_key_lengths = process_sequence(memory_keys)
    memory_values, memory_value_lengths = process_sequence(memory_values)

    distractors = [
        numberize(distractor)
        for distractor in distractors
    ]

    return context, response, memory_keys, memory_key_lengths, memory_values, memory_value_lengths, distractors


stemmer = SnowballStemmer("english")


def process_predict_embed(response):
    response = ' '.join(list(map(stemmer.stem, nltk.word_tokenize(response))))
    response = numberize(response)
    return response
