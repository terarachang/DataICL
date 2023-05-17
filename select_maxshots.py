import numpy as np
import argparse
import sys
import json
import pickle as pkl
import itertools
from glob import glob
import os
import pdb
from collections import defaultdict
from utils.selection import *


def get_balanced_subsets(n_class, n_sets, n_shots, train_labels):

    n_per_lb = len(train_labels) // n_class
    k_per_lb = n_shots // n_class

    ids_per_label = []
    for lb in range(n_class):
        ids = np.where(train_labels == lb)[0]
        assert len(ids) == n_per_lb
        ids_per_label.append(ids)

    subset_ids = np.zeros((n_sets, n_shots), dtype=int)
    for i in range(n_sets):
        indices = []
        for lb in range(n_class):
            indices += list(np.random.choice(ids_per_label[lb], k_per_lb, replace=False))
        np.random.shuffle(indices)
        subset_ids[i] = indices

    return subset_ids


def main(args):

    np.random.seed(0)
    args.model = args.model.lower()
    _, train_labels, train_data, _, _, _ = get_train_dev_data('data', args.task, False)
    n_class = len(train_data[0]["options"])
    assert args.n_shots % n_class == 0

    subset_ids = get_balanced_subsets(n_class, args.n_truncate, args.n_shots, train_labels)
    dump_selected_subsets(args.task, args.model, subset_ids, train_data, "MaxShot")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_shots", type=int, required=True)
    parser.add_argument("--model", type=str, default="gpt-j-6b", required=True)
    parser.add_argument("--task", type=str, default="glue-sst2", required=True)
    parser.add_argument("--n_truncate", type=int, default=50)

    args = parser.parse_args()

    main(args)
