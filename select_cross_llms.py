import numpy as np
import argparse
import sys
import json
import pickle as pkl
import itertools
from glob import glob
import os
import pdb
from config.config import OUT_SELECT
from utils.selection import *


def get_shared_ids(args):
    ids1 = set(np.load(os.path.join(OUT_SELECT, f'{args.model1}-{args.task}_common_ids-CondAcc-good.npy')))
    ids2 = set(np.load(os.path.join(OUT_SELECT, f'{args.model2}-{args.task}_common_ids-CondAcc-good.npy')))
    shared_ids = ids1.intersection(ids2)
    n_overlap = len(shared_ids)

    print(f"# Ex overlap between {args.model1} and {args.model2}: {n_overlap}")
    print(shared_ids)
    return np.array(list(shared_ids))


def generate_subsets(args, ids, n_shots=4):
    assert len(ids) == n_shots
    n_subsets = 24

    _, train_labels, train_data, _, _, _ = get_train_dev_data('data', args.task, False)

    subset_ids = np.zeros((n_subsets, n_shots), dtype=int)
    for i, order in enumerate(itertools.permutations(range(n_shots))):
        subset_ids[i] = ids[list(order)]

    print(subset_ids)
    dump_selected_subsets(args.task, args.model1, subset_ids, train_data, "CrossLLM")
    dump_selected_subsets(args.task, args.model2, subset_ids, train_data, "CrossLLM")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="glue-sst2")
    parser.add_argument("--model1", type=str, default="gpt-j-6b")
    parser.add_argument("--model2", type=str, default="opt-13b")
    args = parser.parse_args()

    ids = get_shared_ids(args)
    generate_subsets(args, ids)
