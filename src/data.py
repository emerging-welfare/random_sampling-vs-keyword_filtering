from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
import random
import numpy as np
import re

def read_set(f):
    
    doc_df = pd.read_json(f, orient="records", lines=True)
    # remove lines that contain the urls
    doc_df['text'] = doc_df.text.apply(lambda d: '\n'.join([l for l in d.split('\n') if not (len(l)<200 and 'url' in l and 'http' in l)])) 
    # remove lines that contain the publication date.
    doc_df['text'] = doc_df.text.apply(lambda d: '\n'.join([l for l in d.split('\n') if not (len(l)<100 and re.search(r'\bJanuary\b|\bFebruary\b|\bMarch\b|\bApril\b|\bMay\b|\bJune\b|\bJuly\b|\bAugust\b|\bSeptember\b|\bOctober\b|\bNovember\b|\bDecember\b', l) and re.search(r'\d{4}',l))]))
    
    return doc_df.to_dict('records')

def fold_iterator_sklearn(all_samples, K=10, dev_ratio=0.1, random_seed=1234):
    """yields K tuples of shape (train, dev, test) """
    random.seed(random_seed)
    random.shuffle(all_samples) # initial shuffle
    _all = np.array(all_samples) # convert to numpy for list indexing

    skf = StratifiedKFold(n_splits=K, random_state=random_seed)
    skf.get_n_splits(_all, [ y["label"] for y in _all])

    for train_index, test_index in skf.split(_all, [ y["label"] for y in _all]):
        trn, dev = train_test_split(_all[train_index], test_size=dev_ratio, random_state=random_seed)
        yield (trn, dev, _all[test_index])
    return