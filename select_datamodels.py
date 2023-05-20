import numpy as np
import argparse
import torch
import os
import itertools
from collections import Counter, defaultdict
from tabulate import tabulate
import json
import sys
from tqdm import tqdm
import pdb

from utils.selection import *
from config.config import OUT_DM


def main(args):
    args.model = args.model.lower()

    # setup
    n_labels, n_shots = setup(args.task)
    n_patterns, label_feats, labelfeat2idx = set_label_features(n_labels)

    # get train / dev set
    x_train, train_labels, train_data, _, dev_labels, _ = get_train_dev_data('data', args.task, False)
    n_train = len(train_labels)

    # label_to_location for selection
    train_label_order = [train_labels[i+1] for i in range(0, n_train, int(n_train/n_labels))]
    assert len(train_label_order) == n_labels
    lb2loc = {v:i for i, v in enumerate(train_label_order)} # label pattern to weight positions
    print(train_label_order)
    print(lb2loc)

    # selection for each label pattern
    weight_counts = torch.zeros((n_patterns, n_train, n_shots), dtype=torch.long)

    weights_16x, biases_16x = load_16x_weights(n_patterns, args.task, args.datamodel_dir, args.n_train_sets)
    print("weights.shape", weights_16x[0].shape, "biases.shape", biases_16x[0].shape)

    for i in range(n_patterns):
        weights, biases = weights_16x[i], biases_16x[i]

        label_pattern = label_feats[i]
        masks = create_mask(label_pattern, lb2loc, n_train, n_shots, n_labels)
        weight_counts[i] = get_weight_counts(weights, masks, n_shots)

    # permutaion invariant set
    if n_labels == 2:
        weight_counts_sum = weight_counts[1:-1].sum([0,2]) # filter [0,0,0,0]
    else:
        weight_counts_sum = weight_counts.sum([0,2])

    if args.is_verbose:
        topk = torch.topk(weight_counts_sum, args.useful_size)
        common_ids0 = topk[1]

        print(weight_counts_sum.max(), weight_counts_sum.float().mean(), weight_counts_sum.min())
        print(topk)
        print(train_labels[common_ids0])

    sorted_ids = np.argsort(-weight_counts_sum.numpy())
    common_ids = get_balanced_topN_ids(sorted_ids, train_labels, args.useful_size, n_labels)
    valid_ids = recombine(common_ids, train_labels, n_labels, n_shots)

    print('Truncate...')
    new_ids = truncate(valid_ids, args.n_truncate, len(common_ids))
    selc_acc = []
    for ids in new_ids:
        acc = get_predicted_acc2(ids, train_labels, weights_16x, biases_16x, labelfeat2idx, n_shots, True)
        selc_acc.append(acc)
    selc_acc = np.array(selc_acc)
    print(f"[Predicted by Datamodel] Avg: {selc_acc.mean():.3f}, Min: {selc_acc.min():.3f}\n")

    method = f"Datamodels"
    save_train_ids(args.task, args.model, common_ids, new_ids, method)
    dump_selected_subsets(args.task, args.model, new_ids, train_data, method)


def create_mask(label_pattern, lb2loc, n_train, n_shots, n_labels):
  masks = torch.zeros((n_train, n_shots), dtype=bool)

  n_per_cls = len(masks) // n_labels
  lbloc_pattern = np.array([lb2loc[lb] for lb in label_pattern])
  for pos, (i, j) in enumerate(zip(lbloc_pattern*n_per_cls, (lbloc_pattern+1)*n_per_cls)):
    masks[i:j, pos] = True
  return masks.float()


def get_predicted_acc(train_ids, weights, biases, n_shots):
  correct = 0
  # w in each datamodel (correctness in each test_example)
  for i, (w, b) in enumerate(zip(weights, biases)):
    w = w.view(-1, n_shots)
    pred = w[train_ids, range(n_shots)].sum().item() + b
    if pred > 0: correct += 1
  return correct/len(weights)


def get_weight_counts(weights, masks, n_shots):
  # mask out invalid training examples that don't belong to the current label pattern
  # return [1000, 4]
  weight_counts = (weights > 0).long().sum(0)
  weight_counts = weight_counts.view(-1, n_shots)*masks.long() #(n_train, position)

  return weight_counts


def load_16x_weights(n_patterns, task, datamodel_dir, sub_dir):
    weights_16x, biases_16x = [], []
    for i in range(n_patterns):
      path = os.path.join(datamodel_dir, task, f"{sub_dir}-feat1-{i}")
      _weights = torch.load(os.path.join(path, "weights.pt"))
      weights_16x.append(_weights[:,:-1])
      biases_16x.append(_weights[:, -1])
    return weights_16x, biases_16x
    

def get_predicted_acc2(subset_ids, train_labels, weights_16x, biases_16x, labelfeat2idx, n_shots, print_pattern=False):
    assert len(subset_ids) == n_shots
    label_pattern = train_labels[subset_ids]
    pattern_type = labelfeat2idx[tuple(label_pattern)]
    weights, biases = weights_16x[pattern_type], biases_16x[pattern_type]

    # estimate acc with the datamodel
    acc = get_predicted_acc(subset_ids, weights, biases, n_shots)
    if print_pattern: 
      print(subset_ids, label_pattern, f'{acc:.3f}')
    return acc

def set_label_features(n_labels):
    if n_labels==2:
        label_feats = list(itertools.product([0,1],[0,1],[0,1],[0,1])) 
    else:
        label_feats = list(itertools.permutations(range(n_labels)))
    label_feats = np.array([list(feat) for feat in label_feats])
    labelfeat2idx = {tuple(feat): i for i, feat in enumerate(label_feats)} # {(0, 0, 0, 0): 0, (0, 0, 0, 1): 1}
    n_patterns = len(label_feats)
    return n_patterns, label_feats, labelfeat2idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_verbose", action="store_true")
    parser.add_argument("--model", type=str, default="gpt-j-6b", required=True)
    parser.add_argument("--task", type=str, default="glue-sst2", required=True)
    parser.add_argument("--n_truncate", type=int, default=50)
    parser.add_argument("--n_train_sets", type=int, required=True)
    parser.add_argument("--n_perm", type=int, default=2)
    parser.add_argument("--useful_size", type=int, default=20)

    args = parser.parse_args()
    args.datamodel_dir = os.path.join(OUT_DM, args.model)

    main(args)
