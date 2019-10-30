import nltk
from nltk.stem import SnowballStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from tqdm import tqdm


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
    print('loading: ', filename)
    model = Word2Vec.load(filename)
    embeddings = {}
    for word in tqdm(vocab):
        if word not in model.wv.vocab:
            continue
        embeddings[vocab[word]] = model.wv.get_vector(word).tolist()
    return embeddings


def load_glove_embeddings(filename='data/glove.6B.100d.txt'):
    print('loading: ', filename)
    lines = open(filename).readlines()
    embeddings = {}
    for line in tqdm(lines):
        word = line.split()[0]
        embedding = list(map(float, line.split()[1:]))
        if word in vocab:
            embeddings[vocab[word]] = embedding

    return embeddings


def numberize(inp):
    inp = inp.split()
    result = list(map(lambda k: vocab.get(k, 0), inp))[-160:]
    return result


def process_sequence(seq):
    seq = pad_sequences([numberize(x) for x in seq], padding='post')
    seq_lens = (seq > 0).sum(axis=-1)
    return seq, seq_lens


def process_train_batch(rows):
    cs, _ = process_sequence([(x['c']) for x in rows])
    rs, _ = process_sequence([(x['r']) for x in rows])
    ys = [int(x['y']) for x in rows]
    memory_keys = []
    memory_values = []
    for row in rows:
        row_keys, row_values = [], []
        keys = list(row['m'].keys())[:5]
        for k in keys:
            v = row['m'][k]
            row_keys += v
            row_values += ([k]*len(v))
        if len(row_keys) < 50:
            row_keys += ['<pad>'] * (50-len(row_keys))
            row_values += ['<pad>'] * (50-len(row_values))
        memory_keys += row_keys
        memory_values += row_values
    memory_keys, memory_key_lengths = process_sequence(memory_keys)
    memory_values, memory_value_lengths = process_sequence(memory_values)

    bsz = len(rows)
    memory_keys = memory_keys.reshape((bsz, 50, -1))
    memory_values = memory_values.reshape((bsz, 50, -1))
    memory_key_lengths = memory_key_lengths.reshape((bsz, 50))
    memory_value_lengths = memory_value_lengths.reshape((bsz, 50))

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


stemmer = SnowballStemmer("english")


def process_predict_embed(response):
    response = ' '.join(list(map(stemmer.stem, nltk.word_tokenize(response))))
    response = numberize(response)
    return response
