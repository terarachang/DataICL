import numpy as np
import argparse
import sys
import json
import pickle as pkl
import itertools
from glob import glob
import torch
import os
import pdb
from collections import defaultdict
from utils.selection import *
from config.config import OUT_SELECT


def main(args):
    args.model = args.model.lower()
    n_labels, n_shots = setup(args.task)

    _, train_labels, train_data, _, _, _ = get_train_dev_data('data', args.task, False)

    fn_best = os.path.join(OUT_SELECT, f'{args.model}-{args.task}_subset_ids-Best{args.n_top}.npy')

    ids = np.load(fn_best)
    common_ids = np.unique(ids.reshape(-1))
    print(ids.shape, len(common_ids))
    valid_ids = recombine(common_ids, train_labels, n_labels, n_shots)
    new_ids = truncate(valid_ids, args.n_truncate, len(common_ids))

    method = f"TopPrompts-{args.n_top}"
    dump_selected_subsets(args.task, args.model, new_ids, train_data, method)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-j-6b", required=True)
    parser.add_argument("--task", type=str, default="glue-sst2", required=True)
    parser.add_argument("--n_top", type=int, default=5)
    parser.add_argument("--n_truncate", type=int, default=50)
    parser.add_argument("--n_train_sets", type=int, default=None)
    parser.add_argument("--n_perm", type=int, default=2)

    args = parser.parse_args()

    main(args)
