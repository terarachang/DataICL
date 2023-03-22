import torch
import os
import numpy as np
import sys
import pdb
#from scipy.stats import rankdata
import argparse
from utils.selection import *
from config.config import OUT_ONESHOT

def main(args):
    args.model = args.model.lower()
    n_labels, n_shots = setup(args.task)

    ckpt_dir = os.path.join(OUT_ONESHOT, args.task, args.model)
    logits = torch.load(os.path.join(ckpt_dir, 'dev-logits.pt'))
    print(logits.shape)
    x_train, train_labels, train_data, _, dev_labels, _ = get_train_dev_data('data', args.task, args.is_unlabel)

    train_ex_acc = (logits.argmax(-1).numpy() == dev_labels).mean(-1)   #[n_train_subsets]
    print(f"[K=1 Acc] Max: {train_ex_acc.max():.3f}, Avg: {train_ex_acc.mean():.3f}, Min: {train_ex_acc.min():.3f}")

    sorted_k1_ids = (-train_ex_acc).argsort()
    #ranks = rankdata(train_ex_acc, method='max')

    topN_ids = get_balanced_topN_ids(sorted_k1_ids, train_labels, args.useful_size, n_labels)
    print(train_ex_acc[topN_ids])

    valid_ids = recombine(topN_ids, train_labels, n_labels, n_shots)
    new_ids = truncate(valid_ids, args.n_truncate, len(topN_ids))

    tag = "-unlabeled" if args.is_unlabel else ""
    method = f"OneShot{tag}"
    dump_selected_subsets(args.task, args.model, new_ids, train_data, method)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--is_unlabel", action="store_true")
    parser.add_argument("--model", type=str, default="gpt-j-6b", required=True)
    parser.add_argument("--task", type=str, default="glue-sst2", required=True)
    parser.add_argument("--n_truncate", type=int, default=50)
    parser.add_argument("--useful_size", type=int, default=20)

    args = parser.parse_args()

    main(args)
