import nltk
from nltk.stem import SnowballStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from tqdm import tqdm
from data import vocab


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




stemmer = SnowballStemmer("english")


def process_predict_embed(response):
    response = ' '.join(list(map(stemmer.stem, nltk.word_tokenize(response))))
    response = numberize(response)
    return response
