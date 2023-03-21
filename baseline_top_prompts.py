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

def get_best_ids(args, dev_labels, tag):
    ic_data = ICData(args.task, args.ckpt_dir, args.n_perm, n_train_sets=None, is_unlabel=args.is_unlabel)
    acc_scores = (ic_data.logits.argmax(-1).numpy() == dev_labels).mean(-1) #[2, n_train_subsets]
    acc_scores = acc_scores.reshape(-1)

    all_permutations = list(itertools.permutations(list(range(args.n_shots))))
    train_ids, permute_ids = flatten_permutation_dimensions(ic_data.train_ids, ic_data.permute_ids, args.n_perm)
    assert len(acc_scores) == len(train_ids) == len(permute_ids)

    sorted_subset_idx = np.argsort(-acc_scores)[:args.n_top]
    print("Top Acc:", acc_scores[sorted_subset_idx])

    best_subset_ids = np.zeros((args.n_top, args.n_shots), dtype=int)
    for i, idx in enumerate(sorted_subset_idx):
        subset_train_ids, order = train_ids[idx], all_permutations[permute_ids[idx]]
        best_subset_ids[i] = subset_train_ids[list(order)] # recover the original order

    np.save(os.path.join(OUT_SELECT, f'{args.model}-{args.task}_subset_ids-Best{args.n_top}{tag}.npy'), best_subset_ids)
    
    return best_subset_ids


def main(args):
    args.model = args.model.lower()
    n_labels, args.n_shots = setup(args.task)
    tag = "-unlabeled" if args.is_unlabel else ""

    _, train_labels, train_data, _, dev_labels, _ = get_train_dev_data('data', args.task, args.is_unlabel)

    ids = get_best_ids(args, dev_labels, tag)
    common_ids = np.unique(ids.reshape(-1))
    print(ids.shape, len(common_ids))
    valid_ids = recombine(common_ids, train_labels, n_labels, args.n_shots)
    new_ids = truncate(valid_ids, args.n_truncate, len(common_ids))

    method = f"TopPrompts-{args.n_top}{tag}"
    dump_selected_subsets(args.task, args.model, new_ids, train_data, method)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_unlabel", action="store_true")
    parser.add_argument("--model", type=str, default="gpt-j-6b", required=True)
    parser.add_argument("--task", type=str, default="glue-sst2", required=True)
    parser.add_argument("--ckpt_dir", type=str, default="Dicl/gpt-j-6b/label_glue-sst2", required=True)
    parser.add_argument("--n_top", type=int, default=5)
    parser.add_argument("--n_truncate", type=int, default=50)
    parser.add_argument("--n_train_sets", type=int, default=None)
    parser.add_argument("--n_perm", type=int, default=2)

    args = parser.parse_args()

    main(args)
