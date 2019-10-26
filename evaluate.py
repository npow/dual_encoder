import data
import numpy as np
import preprocessing
import torch
from tqdm import tqdm

from torch.autograd import Variable

def evaluate(model, size=None, split='dev'):
    model = model.eval()
    """
    Evaluate the model on a subset of dataset.
    """
    if split == 'dev':
        ds = data.get_validation(size)
    else:
        ds = data.get_test(size)
    ds = list(map(preprocessing.process_valid, ds))
    recall_k = {k:0 for k in range(1,11)}

    for e in tqdm(ds):
        context, response, distractors = e
        
        with torch.no_grad():
            cs = Variable(torch.stack([torch.LongTensor(context) for i in range(10)], 0)).cuda()
            rs = [torch.LongTensor(response)]
            rs += [torch.LongTensor(distractor) for distractor in distractors]
            rs = Variable(torch.stack(rs, 0)).cuda()
            
            results, responses = model(cs, rs, [context for i in range(10)])
        results = np.array([e.item() for e in results])

        ranking = np.argsort(-results)
        for k in recall_k.keys():
            k = int(k)
            if 0 in ranking[:k]:
                recall_k[k] += 1

    for k, v in recall_k.items():
        recall_k[k] = v / len(ds)

    return recall_k

if __name__ == '__main__':
    model = torch.load("SAVED_MODEL")
    model.cuda()
    print(evaluate(model))
