import numpy as np
from glob import glob
import argparse
from glob import glob
import torch
import torch.nn as nn
from tqdm import tqdm
import pdb
import json
from scipy import stats
import os
import itertools
import sys
from torch.utils.data import DataLoader, TensorDataset
from utils.selection import *
from train_datamodels import get_indicator, get_loader, get_label_pattern, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    dev_labels = get_labels(args.task, os.path.join('data', args.task), 'dev')
    n_dev = len(dev_labels)

    ic_data = ICData(args.task, args.ckpt_dir, args.n_perm, args.n_test_sets, False)
    datamodel_dir = args.datamodel_dir

    path = os.path.join(datamodel_dir, 'weights.pt')
    pretrained_weights = torch.load(path)
    print("Init. from", path)


    indicator_len = 3*999+1 if args.task in ['glue-mnli', 'scicite'] else 4*1000+1 # +1 for bias term
    for dev_id in tqdm(range(n_dev)):
        test_loader = prep_data(ic_data, args.task, args.feat_type, dev_id, dev_labels[dev_id], indicator_len-1)

        if dev_id == 0: # init.
            l1_losses = torch.zeros(n_dev)

            n_test = len(test_loader.dataset)
            tgts_test = torch.zeros((n_dev, n_test))
            preds_test = tgts_test.clone()
            if args.is_verbose:
                print(f'Test id = {dev_id}')
                print("# Features:", indicator_len)
                print("# Test:", n_test)

        tgts_test[dev_id] = test_loader.dataset.tensors[1].squeeze()

        model = build_model(indicator_len, pretrained_weights[dev_id].unsqueeze(1))
        
        # eval
        preds, l1_ts = evaluate(model, test_loader)
        preds_test[dev_id] = preds.squeeze()

        if args.is_verbose:
            print(f"[Dev-{dev_id}] Average L1 [Test]: {l1_ts:.3f}")

        del model

    torch.save(preds_test, os.path.join(datamodel_dir, "preds_test.pt"))
    torch.save(tgts_test, os.path.join(datamodel_dir, "tgts_test.pt"))

    print('='*50)
    ckpt_basename = os.path.basename(datamodel_dir).split("-")[0]
    print(f"[{ckpt_basename}] Avg L1 of feat-{args.feat_type}", torch.abs(preds_test - tgts_test).mean().item())
    print(f"[{ckpt_basename}] Avg Corr of feat-{args.feat_type}", get_corr(preds_test, tgts_test))
    print('='*50)


def get_corr(preds, tgts):
    # preds_unseen: [n_dev, n_unseen]
    corr = []
    for pred, tgt in zip(preds, tgts):
        corr.append(stats.pearsonr(pred, tgt)[0])

    corr = np.array(corr)

    return np.round(corr.mean(), 3) # avg over all datamodels (dev_examples)


def prep_data(ic_data, task, feat_type, dev_id, label, bias_id):
    
    k = 3 if task in ["glue-mnli", "scicite"] else 4
    all_permute = list(itertools.permutations(list(range(k))))
    n_test_sets, n_labels, n_perm = ic_data.n_train_sets, ic_data.n_labels, ic_data.n_perm

    def get_datamodel_target(i):
        # get the target of the currenct datamodel (dev_id)
        Y = ic_data.logits[i][:, dev_id]       # [n_test_sets, n_labels]
        masked_y = Y.clone()
        masked_y[:, label] = -1000.
        largest_wrong_logits = torch.max(masked_y, 1)[0]
        Y = Y[:, label] - largest_wrong_logits  # logits(corr) - logits(incorr)
        Y = Y.unsqueeze(-1)                     # [n_test_sets, 1]
        assert len(Y) == n_test_sets
        return Y


    test_X, test_Y = [], []
    test_label_patterns = []
    # for each permutation
    for i in range(n_perm):
        Y = get_datamodel_target(i)
        permute_i = ic_data.permute_ids[:, i]

        # preprocessing: get the valid examples in the current label patterns
        label_patterns = torch.zeros(len(Y), dtype=int)
        for j, (lbs, p_i) in enumerate(zip(ic_data.all_prompt_labels, permute_i)):
            # for each training subset
            order = all_permute[p_i]
            label_patterns[j] = get_label_pattern(n_labels, lbs, order)

        label_t = int(feat_type.split('-')[1])
        curr_lb_ids = torch.nonzero(label_patterns == label_t, as_tuple=True)[0]
        valid_ts_ids = curr_lb_ids.tolist()
        valid_ts_ids = sorted(valid_ts_ids)


        X = torch.zeros((n_test_sets, k+1), dtype=torch.long)
        X[:, -1] = bias_id
        for j, (lbs, p_i) in enumerate(zip(ic_data.all_prompt_labels, permute_i)):
            order = all_permute[p_i]
            X[j][:k] = get_indicator(ic_data.train_ids[j], order, k)

        test_X.append(X[valid_ts_ids])
        test_Y.append(Y[valid_ts_ids])
        test_label_patterns.extend(label_patterns[valid_ts_ids])

    # create dataloaders
    test_X = torch.cat(test_X, 0)
    test_Y = torch.cat(test_Y, 0)
    assert len(test_X) == len(test_Y), "train X, Y sizes mismatch"

    test_loader = get_loader(test_X, test_Y, False)

    return test_loader


def build_model(indicator_len, weights=None):
    model = nn.Embedding(indicator_len, 1).from_pretrained(weights, freeze=False)
    model.to(device)
    return model



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_verbose", action="store_true")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--datamodel_dir", type=str, required=True)
    parser.add_argument("--n_test_sets", type=int, default=5000)
    parser.add_argument("--feat_type", type=str, required=True)
    parser.add_argument("--n_perm", type=int, default=1)

    args = parser.parse_args()

    main(args)
