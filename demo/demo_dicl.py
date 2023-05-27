import numpy as np
import argparse
import json
import itertools
import torch
import os
import pdb
from utils.selection import *


def get_aligned_data(args, dev_labels, train_data):
    ic_data = ICData(args.task, args.logits_dir, args.n_perm, n_train_sets=None, is_unlabel=args.is_unlabel)
    # ic_data.logits.shape: [2 permutations, n_combinations, n_dev_examples, n_labels]
    flat_logits = ic_data.logits.view(-1, *ic_data.logits.shape[2:])
    acc_scores = (flat_logits.argmax(-1).numpy() == dev_labels).mean(-1)

    all_permutations = list(itertools.permutations(list(range(args.n_shots))))
    train_ids, permute_ids = flatten_permutation_dimensions(ic_data.train_ids, ic_data.permute_ids, args.n_perm)
    assert len(flat_logits) == len(acc_scores) == len(train_ids) == len(permute_ids)

    data = []
    for logits, acc, k_ids, p in zip(flat_logits, acc_scores, train_ids, permute_ids):
        order = list(all_permutations[p])
        k_ids = k_ids[order] # recover the original order
        k_texts = [train_data[i] for i in k_ids]
        dp = {'train_ids': k_ids, 'train_examples': k_texts, 'dev accuracy': acc, 'logits': logits}
        data.append(dp)
    print("len(data):", len(data))

    if args.verbose:
        print('-'*100)
        import pprint
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(data[0])
        print('-'*100)

    return data


def main(args):
    args.model = args.model.lower()
    n_labels, args.n_shots = setup(args.task)

    _, train_labels, train_data, _, dev_labels, _ = get_train_dev_data('data', args.task, args.is_unlabel)

    data = get_aligned_data(args, dev_labels, train_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_unlabel", action="store_true", help="unlabeled setup")
    parser.add_argument("--verbose", action="store_true", help="print a datapoint")
    parser.add_argument("--model", type=str, default="gpt-j-6b", required=True)
    parser.add_argument("--task", type=str, default="glue-sst2", required=True)
    parser.add_argument("--n_perm", type=int, default=2, help="number of permutations of K in-context examples")

    args = parser.parse_args()
    args.logits_dir = os.path.join("Dicl", args.model, f"{'unlabel' if args.is_unlabel else 'label'}_{args.task}")
    print(args)

    main(args)
