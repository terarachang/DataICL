import numpy as np
import json
import pdb
import pickle
import os
from glob import glob
from collections import defaultdict
from tabulate import tabulate
from transformers import AutoTokenizer

def get_data(task, split='train'):
    fn = f"data/{task}/{task}_500_0_{split}.jsonl"
    data, labels = [], []
    with open(fn, "r") as f:
        for line in f:
            dp = json.loads(line)
            data.append(dp)
            labels.append(dp['options'].index(dp['output']))
    return data, np.array(labels)

def prep_text(task, data):
    def strip_template(text):
        if task == 'glue-sst2':
            return text.split("Review: ",1)[1].rsplit("\nSentiment:", 1)[0].strip()
        elif task == 'subj':
            return text.split("Input: ",1)[1].rsplit("\nType:", 1)[0].strip()
        elif task == 'ag_news':
            return text.split("Article: ",1)[1].rsplit("\nAnswer:", 1)[0].strip()
        elif task == 'scicite':
            seg1 = "Is the following citation from a scientific paper describing a method, a result, or background?\n"
            seg2 = "\nAnswer:"
            return text.split(seg1,1)[1].rsplit(seg2, 1)[0].strip('"')
        elif task == 'boolq':
            seg1 = "Exercise: read the text and answer the question by yes or no.\n\nText: "
            return text.split(seg1,1)[1].strip()
        
    texts = []
    for dp in data:
        txt = dp['input']
        texts.append(strip_template(txt))
    return texts


'''
def get_identified_ids(model, task, is_good):
    if is_good:
        ids_dir = "final_results/meta/"
        tag = "highestAvg"
    else:
        ids_dir = "final_results/meta_lowest/"
        tag = "lowestAvg"
    fn = sorted(glob(os.path.join(ids_dir, f'{model}-{task}_common_ids-{tag}*s20.npy')))

    if len(fn) != 1:
        print(fn)
        #pdb.set_trace()

    print(fn[-1])
    simple_ids = np.load(fn[-1])
    return set(simple_ids)

def get_len(data, tokenizer, model):
    lengths = np.zeros(len(data), dtype=int)
    for i, txt in enumerate(data):
        enc = tokenizer.encode(txt)
        lengths[i] = len(enc)

    if 'opt-' in model: # not counting bos
        lengths = lengths -1
    return lengths

def nltk_tokenize_words(data):
    from nltk.tokenize import word_tokenize

    tokenized_data = []
    for txt in data:
        tokenized_data.append(word_tokenize(txt))
    return tokenized_data

def get_distinct_ngram(tokenized_data, n):
    from nltk.util import ngrams

    distinct_set = set()
    n_total = 0
    for i, txt in enumerate(tokenized_data):
        cur_ngram = set(list(ngrams(txt, n)))
        distinct_set.update(cur_ngram)
        n_total += len(txt)

    return len(distinct_set) / n_total
'''
