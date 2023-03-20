import os
import csv
import json
import string
import numpy as np
import random
import torch
import pdb

def load_data(split, n, seed, dataset, template_dir=""):

    data_path = os.path.join("data", dataset, template_dir,
                             "{}_{}_{}_{}.jsonl".format(dataset, n, seed, split))
    data = []
    with open(data_path, "r") as f:
        for line in f:
            dp = json.loads(line)
            data.append(dp)
    return data


def random_subset_of_comb(iterable, r, n_sets):
    ''' for binary tasks '''

    def random_combination(iterable, r):
        # "Random selection from combinations(iterable, r)"
        pool = tuple(iterable)
        n = len(pool)
        indices = sorted(random.sample(range(n), r))
        return tuple(pool[i] for i in indices)

    results = set()
    while True:
        new_combo = random_combination(iterable, r)

        if new_combo not in results:
            results.add(new_combo)

        if len(results) >= n_sets:
            break

    return np.array(list(results))


def balanced_subset_of_comb(iterable, r, n_sets, train_labels):
    ''' for multiclass tasks '''

    pool = tuple(iterable)
    n_per_lb = len(iterable) // r

    def unique(array):
        uniq, index = np.unique(array, return_index=True)
        return uniq[index.argsort()]
    labels = unique(train_labels)

    assert len(labels) == r, f"#labels:{len(labels)}, #demons:{r}"
    ids_per_label = []
    for lb in labels:
        ids = np.where(train_labels == lb)[0]
        assert len(ids) == n_per_lb
        ids_per_label.append(ids)
    #print(ids_per_label)

    results = set()
    while True:
        indices = random.sample(range(n_per_lb), r)
        indices = np.array([ids_per_label[l][j] for l, j in enumerate(indices)])

        new_combo = tuple(pool[i] for i in indices)

        if new_combo not in results:
            results.add(new_combo)

        if len(results) >= n_sets:
            break

    return np.array(list(results))
