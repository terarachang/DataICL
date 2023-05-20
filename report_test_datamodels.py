import numpy as np
import os
from glob import glob
import torch
import pdb
from tabulate import tabulate
from scipy import stats
import argparse


def get_corr(preds, tgts):
    # preds_unseen: [n_dev, n_unseen]
    corr = []
    for pred, tgt in zip(preds, tgts):
        corr.append(stats.pearsonr(pred, tgt)[0])

    corr = np.array(corr)
    return corr.mean() # avg over all datamodels (dev_examples)


def get_results(size, n_patterns, task, model, out_dir):
    l1_tr, l1_ts = np.zeros(n_patterns), np.zeros(n_patterns)
    corr_tr, corr_ts = np.zeros(n_patterns), np.zeros(n_patterns)

    all_preds_ts, all_preds_tr, all_tgts_ts, all_tgts_tr = [], [], [], []
    for i in range(n_patterns):
        d = os.path.join(out_dir, f'{size}-feat1-{i}')
        preds_test =  torch.load(os.path.join(d, 'preds_test.pt'))
        preds_train = torch.load(os.path.join(d, 'preds_train.pt'))
        tgts_test = torch.load(os.path.join(d, 'tgts_test.pt'))
        tgts_train = torch.load(os.path.join(d, 'tgts_train.pt'))

        all_preds_ts.append(preds_test)
        all_preds_tr.append(preds_train)
        all_tgts_ts.append(tgts_test)
        all_tgts_tr.append(tgts_train)

    all_preds_ts = torch.cat(all_preds_ts, 1)
    all_preds_tr = torch.cat(all_preds_tr, 1)
    all_tgts_ts = torch.cat(all_tgts_ts, 1)
    all_tgts_tr = torch.cat(all_tgts_tr, 1)

    l1_tr = torch.abs(all_tgts_tr - all_preds_tr).mean().item()
    l1_ts = torch.abs(all_tgts_ts - all_preds_ts).mean().item()

    corr_tr = get_corr(all_preds_tr, all_tgts_tr)
    corr_ts = get_corr(all_preds_ts, all_tgts_ts)

    results = {"l1_train": round(l1_tr, 3), "l1_test": round(l1_ts, 3), 
        "corr_train": round(corr_tr, 3), "corr_test": round(corr_ts, 3)}
    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--datamodel_dir", type=str, required=True)
    parser.add_argument("--n_patterns", type=int, required=None)
    parser.add_argument("--n_train_sets", type=int, help="the size of the training set to train datamodels")
    args = parser.parse_args()
    print(args)

    out_dir = os.path.join(args.datamodel_dir, args.task)

    results = get_results(args.n_train_sets, args.n_patterns, args.task, args.model, out_dir)
    print(results)
