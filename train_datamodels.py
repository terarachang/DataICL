import numpy as np
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
from utils.selection import ICData, get_labels


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    dev_labels = get_labels(args.task, os.path.join('data', args.task), 'dev')
    n_dev = len(dev_labels)

    ic_data = ICData(args.task, args.ckpt_dir, args.n_perm, args.n_train_sets, False)
    n_train_sets = ic_data.n_train_sets
    datamodel_dir = os.path.join(args.datamodel_dir, args.task, n_train_sets)

    if args.do_init:
        path = os.path.join(datamodel_dir, 'weights.pt')
        pretrained_weights = torch.load(path)
        print("Init. from", path)


    indicator_len = 3*999+1 if args.task in ['glue-mnli', 'scicite'] else 4*1000+1 # +1 for bias term
    for dev_id in tqdm(range(n_dev)):
        print(f'Test id = {dev_id}')
        train_loader, train_unshf_loader = \
            prep_data(ic_data, args.task, args.feat_type, dev_id, dev_labels[dev_id], indicator_len-1)

        if dev_id == 0: # init.
            weights = torch.zeros((n_dev, indicator_len))
            l1_losses = torch.zeros(n_dev)

            n_seen = len(train_loader.dataset)
            tgts_train = torch.zeros((n_dev, n_seen))
            preds_train = tgts_train.clone()
            print("# Features:", indicator_len)
            print("# Train:", n_seen)

        tgts_train[dev_id] = train_unshf_loader.dataset.tensors[1].squeeze()

        if args.do_init:
            model, optimizer = build_model(indicator_len, pretrained_weights[dev_id].unsqueeze(1))
        else:
            model, optimizer = build_model(indicator_len)
        
        loss_fn = nn.MSELoss()

        # start training
        model.train()
        for e in range(50):
            for indicators, tgts in train_loader:
                indicators, tgts = indicators.to(device), tgts.to(device)

                optimizer.zero_grad()
                preds = model(indicators).sum(1)
                loss = loss_fn(preds, tgts)
                loss.backward()
                optimizer.step()

            if args.is_verbose and e % 20 == 0:
                print(loss.item())

        # eval on train set; make sure it is well fitted
        weights[dev_id] = model.weight.squeeze().detach().cpu()

        preds, l1_tr = evaluate(model, train_unshf_loader)
        preds_train[dev_id] = preds.squeeze()

        print(f"Average L1 [Train]: {l1_tr:.3f}")

        del model, optimizer
        print('-'*50)

    if 'feat1-' in args.feat_type:
        datamodel_dir += f"-{args.feat_type}"
    if not os.path.exists(datamodel_dir):
        os.makedirs(datamodel_dir)

    torch.save(weights, os.path.join(datamodel_dir, "weights.pt"))
    torch.save(preds_train, os.path.join(datamodel_dir, "preds_train.pt"))
    torch.save(tgts_train, os.path.join(datamodel_dir, "tgts_train.pt"))


def get_label_pattern(n_labels, prompt_labels, order):
    if n_labels==2:
        label_feats = list(itertools.product([0,1],[0,1],[0,1],[0,1]))
    else:
        label_feats = list(itertools.permutations(range(n_labels)))
    label_feats = {v:i for i,v in enumerate(label_feats)}

    permuted_prompt_labels = tuple(prompt_labels[list(order)]) 
    ind_i = label_feats[permuted_prompt_labels]
    return ind_i

def get_indicator(example_ids, order, k):
    assert len(order) == len(example_ids) == k
    vector = example_ids*k + order
    return torch.LongTensor(vector)

def get_loader(inps, tgts, is_train):
    dataset = TensorDataset(inps, tgts)
    loader = DataLoader(dataset, batch_size=512, shuffle=is_train)
    return loader


def prep_data(ic_data, task, feat_type, dev_id, label, bias_id):
    
    k = 3 if task in ["glue-mnli", "scicite"] else 4
    all_permute = list(itertools.permutations(list(range(k))))
    n_train_sets, n_labels, n_perm = ic_data.n_train_sets, ic_data.n_labels, ic_data.n_perm

    def get_datamodel_target(i):
        # get the target of the currenct datamodel (dev_id)
        Y = ic_data.logits[i][:, dev_id]       # [n_train_sets, n_labels]
        print("logits.shape", Y.shape)
        masked_y = Y.clone()
        masked_y[:, label] = -1000.
        largest_wrong_logits = torch.max(masked_y, 1)[0]
        Y = Y[:, label] - largest_wrong_logits  # logits(corr) - logits(incorr)
        Y = Y.unsqueeze(-1)                     # [n_train_sets, 1]
        assert len(Y) == n_train_sets
        return Y


    train_X, train_Y = [], []
    train_label_patterns = []
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

        if '-' in feat_type: # only focus on a single label pattern
            label_t = int(feat_type.split('-')[1])
            curr_lb_ids = torch.nonzero(label_patterns == label_t, as_tuple=True)[0]
            valid_tr_ids = curr_lb_ids.tolist()
            valid_tr_ids = sorted(valid_tr_ids)
        else:                # use all data => flatten
            valid_tr_ids = list(range(n_train_sets))


        X = torch.zeros((n_train_sets, k+1), dtype=torch.long)
        X[:, -1] = bias_id
        for j, (lbs, p_i) in enumerate(zip(ic_data.all_prompt_labels, permute_i)):
            order = all_permute[p_i]
            X[j][:k] = get_indicator(ic_data.train_ids[j], order, k)

        # Note: split inside the Permutation loop to avoid Combination leackage
        # train split
        train_X.append(X[valid_tr_ids])
        train_Y.append(Y[valid_tr_ids])
        train_label_patterns.extend(label_patterns[valid_tr_ids])

    # create dataloaders
    train_X = torch.cat(train_X, 0)
    train_Y = torch.cat(train_Y, 0)
    print("X:", train_X.shape, "; Y:", train_Y.shape)
    assert len(train_X) == len(train_Y), "train X, Y sizes mismatch"

    train_loader = get_loader(train_X, train_Y, True)
    train_unshf_loader = get_loader(train_X, train_Y, False)

    return train_loader, train_unshf_loader


@torch.no_grad()
def evaluate(model, data_loader):
    model.eval()
    L1 = 0
    all_preds = []
    for indicators, tgts in data_loader:
        indicators = indicators.to(device)
        preds = model(indicators).sum(1).cpu()

        L1 += torch.abs(preds - tgts).sum().item()
        all_preds.append(preds)

    avg_l1 = round(L1 / len(data_loader.dataset), 3)
    return torch.cat(all_preds), avg_l1


def build_model(indicator_len, weights=None):
    if weights is None:
        weights = nn.Linear(indicator_len, 1, bias=False).weight.view((-1, 1))
    model = nn.Embedding(indicator_len, 1).from_pretrained(weights, freeze=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
    model.to(device)
    return model, optimizer



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_verbose", action="store_false")
    parser.add_argument("--do_init", action="store_true")
    parser.add_argument("--task", type=str, default="glue-sst2", required=True)
    parser.add_argument("--ckpt_dir", type=str, default="Dicl/gpt-j-6b/label_sst2", required=True)
    parser.add_argument("--datamodel_dir", type=str, default="out_datamodel")
    parser.add_argument("--n_train_sets", type=int, default=None)
    parser.add_argument("--feat_type", type=str, default="feat1-0", required=True)
    parser.add_argument("--n_perm", type=int, default=2)

    args = parser.parse_args()

    main(args)
