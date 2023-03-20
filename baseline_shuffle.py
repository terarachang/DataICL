import numpy as np
import sys
import json
import pickle as pkl
import itertools
from glob import glob
import torch
import os
import pdb
import argparse
from collections import defaultdict
from utils.selection import *
from config.config import OUT_SELECT


def main(args):
    args.model = args.model.lower()
    n_labels, args.n_shots = setup(args.task)

    ic_data = ICData(args.task, args.ckpt_dir, args.n_perm, n_train_sets=None, is_unlabel=args.is_unlabel)
    n_train_sets = ic_data.n_train_sets
    assert n_labels == ic_data.n_labels

    x_train, train_labels, train_data, _, dev_labels, _ = get_train_dev_data('data', args.task, args.is_unlabel)

    acc_scores = (ic_data.logits.argmax(-1).numpy() == dev_labels).mean(-1) #[2, n_train_subsets]
    print(acc_scores.shape)

    shuffle_best_k(args, acc_scores, train_data, ic_data.train_ids, ic_data.permute_ids)


def shuffle_best_k(args, acc_scores, train_data, train_ids, permute_ids, k_best=10):

    def shuffle_subsets(best_ids):
        best_ids = best_ids.copy()
        n_shuffles = args.n_truncate // k_best
        all_permute = np.array(list(itertools.permutations(list(range(args.n_shots)))))[1:] # no original order

        np.random.seed(0)
        shulffled_best_ids = np.zeros((k_best, n_shuffles, args.n_shots), dtype=int)
        for i, ids in enumerate(best_ids):
            shuf_orders = all_permute[np.random.choice(len(all_permute), n_shuffles, replace=False)]
            shulffled_best_ids[i] = ids[shuf_orders] # [n samples, k orders]

        return shulffled_best_ids.reshape(-1, args.n_shots)

    # get best_ids
    all_permutations = list(itertools.permutations(list(range(args.n_shots))))
    acc_scores = acc_scores.reshape(-1)
    train_ids, permute_ids = flatten_permutation_dimensions(train_ids, permute_ids, args.n_perm)
    assert len(acc_scores) == len(train_ids) == len(permute_ids)

    sorted_subset_idx = np.argsort(-acc_scores)[:k_best]
    print("Top Acc:", acc_scores[sorted_subset_idx])

    best_subset_ids = np.zeros((k_best, args.n_shots), dtype=int)
    for i, idx in enumerate(sorted_subset_idx):
        subset_train_ids, order = train_ids[idx], all_permutations[permute_ids[idx]]
        best_subset_ids[i] = subset_train_ids[list(order)] # recover the original order
    np.save(os.path.join(OUT_SELECT, f'{args.model}-{args.task}_subset_ids-Best{k_best}.npy'), best_subset_ids)

    shuffled_subset_ids = shuffle_subsets(best_subset_ids)
    best_and_shuffled_ids = np.concatenate((best_subset_ids, shuffled_subset_ids))
    print(best_and_shuffled_ids.shape)

    tag = "-unlabeled" if args.is_unlabel else ""
    method = f"Shuffle{tag}"
    dump_selected_subsets(args.task, args.model, best_and_shuffled_ids, train_data, method)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--is_unlabel", action="store_true")
    parser.add_argument("--model", type=str, default="gpt-j-6b", required=True)
    parser.add_argument("--task", type=str, default="glue-sst2", required=True)
    parser.add_argument("--ckpt_dir", type=str, default="Dicl/gpt-j-6b/label_glue-sst2", required=True)
    parser.add_argument("--n_truncate", type=int, default=30)
    parser.add_argument("--n_perm", type=int, default=2)

    args = parser.parse_args()

    main(args)
