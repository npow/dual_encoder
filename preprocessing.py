import nltk
from nltk.stem import SnowballStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec


def load_vocab(filename):
    lines = open(filename).readlines()
    return {
        word.strip(): i
        for i, word in enumerate(lines)
    }


vocab = load_vocab('data/vocabulary.txt')


def load_embeddings(vectors):
    if vectors == 'glove':
        return load_glove_embeddings()
    elif vectors == 'stackexchange':
        return load_stackexchange_embeddings()
    else:
        raise 'Unknown embeddings: {}'.format(vectors)


def load_stackexchange_embeddings(filename='vectors/word2vec_stackexchange.model'):
    model = Word2Vec.load(filename)
    embeddings = {}
    for word in vocab:
        if word not in model.wv.vocab:
            continue
        embeddings[vocab[word]] = model.wv.get_vector(word).tolist()
    return embeddings


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


def process_train_batch(rows):
    cs = [numberize(x) for x in rows['c']]
    rs = [numberize(x) for x in rows['r']]
    ys = [int(x) for x in rows['y']]
    memory_keys = []
    memory_values = []
    for d in rows['m']:
        for k, v in d.items():
            memory_keys.append(v)
            memory_values.append([k]*len(v))
    memory_keys, memory_key_lengths = process_sequence(memory_keys)
    memory_values, memory_value_lengths = process_sequence(memory_values)
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
        memory_keys.append(v)
        memory_values.append([k]*len(v))
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


stemmer = SnowballStemmer("english")


def process_predict_embed(response):
    response = ' '.join(list(map(stemmer.stem, nltk.word_tokenize(response))))
    response = numberize(response)
    return response
