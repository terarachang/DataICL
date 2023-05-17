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
from config.config import OUT_SCORES
from utils.selection import *


def main(args):
    args.model = args.model.lower()
    n_labels, n_shots = setup(args.task)

    # load data
    ic_data = ICData(args.task, args.ckpt_dir, args.n_perm, args.n_train_sets, args.is_unlabel)
    n_train_sets = ic_data.n_train_sets
    assert n_labels == ic_data.n_labels

    x_train, train_labels, train_data, _, dev_labels, _ = get_train_dev_data('data', args.task, args.is_unlabel)

    # get acc of each training subset; map train_id to acc scores
    acc_scores = (ic_data.logits.argmax(-1).numpy() == dev_labels).mean(-1) #[2, n_train_subsets]
    print(acc_scores.shape)

    id2acc = defaultdict(list)
    for i, subset_ids in enumerate(ic_data.train_ids):
        for idx in subset_ids:
            id2acc[idx].extend(acc_scores[:,i])

    avg_accs = np.zeros(len(train_labels))
    for idx, acc_list in id2acc.items():
        avg_accs[idx] = np.array(acc_list).mean()
    print(f"[Selection; Socres of Train Ex] Max: {avg_accs.max():.3f}, Avg: {avg_accs.mean():.3f}, Min: {avg_accs.min():.3f}")
    os.makedirs(OUT_SCORES, exist_ok=True)
    np.save(os.path.join(OUT_SCORES, f'CondAcc_{args.model}_{args.task}'), avg_accs)

    # sort train_ids and select the training examples with the highest avg acc (CondAcc-good)
    sorted_ids = np.argsort(-avg_accs)
    print("Selected training examples:")
    common_ids = get_balanced_topN_ids(sorted_ids, train_labels, args.useful_size, n_labels)

    if args.is_verbose:
        for (txt, lb) in zip(x_train[common_ids], train_labels[common_ids]):
            txt = " ".join(txt.split('\n'))
            print(txt, lb)
        print(avg_accs[common_ids])

    # sample from P(set_size, n_shots)
    valid_ids = recombine(common_ids, train_labels, n_labels, n_shots)
    new_ids = truncate(valid_ids, args.n_truncate, len(common_ids))

    tag = "-unlabeled" if args.is_unlabel else ""
    method = f"CondAcc-good{tag}"
    save_train_ids(args.task, args.model, common_ids, new_ids, method)
    dump_selected_subsets(args.task, args.model, new_ids, train_data, method)

    # CondAcc-bad: the subset of examples with the lowest avg acc
    if not args.is_unlabel:
        method = f"CondAcc-bad"
        sorted_worst_ids = np.argsort(avg_accs)
        dump_subsets_by_train_ids(args, sorted_worst_ids, train_labels, train_data, n_shots, n_labels, method, True)

    flat_train_ids, flat_permute_ids = flatten_permutation_dimensions(ic_data.train_ids, ic_data.permute_ids, args.n_perm)
    if args.is_unlabel:
        is_topK_gold(args.task, acc_scores, flat_train_ids, flat_permute_ids, n_shots)

    dump_random_subsets(args, train_data, train_labels, flat_train_ids, flat_permute_ids, n_shots, n_labels, tag)


def is_topK_gold(task, acc_scores, train_ids, permute_ids, n_shots):
    all_permutations = list(itertools.permutations(list(range(n_shots))))

    acc_scores = acc_scores.reshape(-1)
    sorted_subset_idx = np.argsort(-acc_scores)[:5] # Top5 subsets
    assert len(train_ids) == len(permute_ids) == len(acc_scores)

    is_groundtruth = np.load(os.path.join('data', task, 'unlabeled', 'is_groundtruth.npy'))
    for idx in sorted_subset_idx:
        subset_train_ids, order = train_ids[idx], all_permutations[permute_ids[idx]]
        subset_train_ids = subset_train_ids[list(order)]
        print(acc_scores[idx], is_groundtruth[subset_train_ids])


def dump_subsets_by_train_ids(args, selc_tr_ids, train_labels, train_data, n_shots, n_labels, method, save_ids=False):
    selc_tr_ids = get_balanced_topN_ids(selc_tr_ids, train_labels, args.useful_size, n_labels)
    valid_ids = recombine(selc_tr_ids, train_labels, n_labels, n_shots, verbose=False)
    subset_ids = truncate(valid_ids, args.n_truncate, args.useful_size)
    dump_selected_subsets(args.task, args.model, subset_ids, train_data, method)
    if save_ids:
        save_train_ids(args.task, args.model, selc_tr_ids, subset_ids, method)


def dump_random_subsets(args, train_data, train_labels, train_ids, permute_ids, n_shots, n_labels, tag):
    ''' the All and Random baseline '''

    all_permutations = list(itertools.permutations(list(range(n_shots))))
    
    # All baseline: sample balanced prompts from the entire training set
    rand_subset_ids = []
    i = 0 
    while True:
        subset_train_ids = train_ids[i]
        if len(np.unique(train_labels[subset_train_ids])) == n_labels:
            order = all_permutations[permute_ids[i]] # recover sampled order
            subset_train_ids = subset_train_ids[list(order)]
            rand_subset_ids.append(subset_train_ids)
        if len(rand_subset_ids) == args.n_truncate: break
        i += 1

    rand_subset_ids = np.stack(rand_subset_ids)
    dump_selected_subsets(args.task, args.model, rand_subset_ids, train_data, f"All{tag}")

    # Random baseline: randomly sample n_useful training ids and recombine them
    if not args.is_unlabel:
        method = f"Random"
        np.random.seed(0)
        rand_tr_ids = np.random.choice(len(train_labels), 100, replace=False)
        dump_subsets_by_train_ids(args, rand_tr_ids, train_labels, train_data, n_shots, n_labels, method)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--is_unlabel", action="store_true")
    parser.add_argument("--is_verbose", action="store_true")
    parser.add_argument("--model", type=str, default="gpt-j-6b", required=True)
    parser.add_argument("--task", type=str, default="glue-sst2", required=True)
    parser.add_argument("--ckpt_dir", type=str, default="Dicl/gpt-j-6b/label_glue-sst2", required=True)
    parser.add_argument("--n_truncate", type=int, default=50)
    parser.add_argument("--n_train_sets", type=int, default=None)
    parser.add_argument("--n_perm", type=int, default=2)
    parser.add_argument("--useful_size", type=int, default=20)

    args = parser.parse_args()

    main(args)
