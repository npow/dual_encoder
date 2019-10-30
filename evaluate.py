import data
import numpy as np
import torch
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences

from torch.autograd import Variable


def evaluate(L, model, size=None):
    model = model.eval()
    """
    Evaluate the model on a subset of dataset.
    """
    recall_k = {k: 0 for k in range(1, 11)}

    for row in tqdm(L):
        context = row['c']
        response = row['r']
        distractors = row['ds']
        memory_keys = row['memory_keys']
        memory_key_lengths = row['memory_key_lengths']
        memory_values = row['memory_values']
        memory_value_lengths = row['memory_value_lengths']

        with torch.no_grad():
            cs = Variable(torch.stack([torch.LongTensor(context) for i in range(10)], 0)).cuda()
            rs = [response] + distractors
            rs = torch.from_numpy(pad_sequences(rs)).long().cuda()

            memory_keys = torch.LongTensor([memory_keys]).cuda()
            memory_key_lengths = torch.LongTensor([memory_key_lengths]).cuda()
            memory_values = torch.LongTensor([memory_values]).cuda()
            memory_value_lengths = torch.LongTensor([memory_value_lengths]).cuda()

            results, responses = model(contexts=cs, responses=rs,
                                       memory_keys=memory_keys, memory_key_lengths=memory_key_lengths,
                                       memory_values=memory_values, memory_value_lengths=memory_value_lengths)
            results = np.array([e.item() for e in results])

        ranking = np.argsort(-results)
        for k in recall_k.keys():
            k = int(k)
            if 0 in ranking[:k]:
                recall_k[k] += 1

    for k, v in recall_k.items():
        recall_k[k] = v / len(L)

    return recall_k


if __name__ == '__main__':
    model = torch.load("SAVED_MODEL")
    model.cuda()
    print(evaluate(model))
