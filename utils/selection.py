import numpy as np
import itertools
import os
import pickle as pkl
import json
import pdb
from glob import glob
import torch
from config.config import OUT_SELECT

os.makedirs(OUT_SELECT, exist_ok=True)


def get_data(task, path, split='dev', n=500):
  texts, labels, data = [], [], []
  fn = f"{path}/{task}_{n}_0_{split}.jsonl"

  with open(fn, "r") as f:
    for line in f:
      dp = json.loads(line)
      texts.append(dp["input"])
      labels.append(dp['options'].index(dp['output']))
      data.append(dp)
  return np.array(texts), np.array(labels).astype('int'), data


def get_labels(task, path, split):
    fn = os.path.join(path, f"{task}_500_0_{split}.jsonl")
    labels = []
    label_words = None
    with open(fn, "r") as f:
        for line in f:
            dp = json.loads(line)
            if label_words == None:
                label_words = dp['options']
            labels.append(label_words.index(dp['output']))
    labels = np.array(labels, dtype='int')
    return labels

def get_train_dev_data(data_dir, task, is_unlabel):
    path = os.path.join(data_dir, task)
    path_train = os.path.join(path, 'unlabeled') if is_unlabel else path
    x_train, train_labels, train_data = get_data(task, path_train, 'train')
    x_dev, dev_labels, dev_data = get_data(task, path, 'dev')

    if is_unlabel: assert len(x_train) > 1000, len(x_train)
    else: assert len(x_train) <= 1000, len(x_train)

    return x_train, train_labels, train_data, x_dev, dev_labels, dev_data


def get_prompt_labels(task, ckpt_dir, train_ids, is_unlabel=False):
    if is_unlabel:
        path_train = os.path.join('data', task, 'unlabeled')
    else:
        path_train = os.path.join('data', task)
    train_labels = get_labels(task, path_train, "train")

    all_prompt_labels = [train_labels[ids] for ids in train_ids]
    return all_prompt_labels

def setup(task):
    if task in ['glue-sst2', 'boolq', 'subj']:
        n_labels, K = 2, 4
    elif task == 'ag_news':
        n_labels, K = 4, 4
    elif task in ['glue-mnli', 'scicite']:
        n_labels, K = 3, 3
    else:
        raise Exception(f"{task} n_labels, K not defined!")
    return n_labels, K


class ICData():
    def __init__(self, task, ckpt_dir, n_perm, n_train_sets=None, is_unlabel=False):
        self.ckpt_dir = ckpt_dir
        self.n_perm = n_perm
        self.n_train_sets = n_train_sets
        self.is_unlabel = is_unlabel

        # set logits, n_labels, n_train_sets
        self.load_target_model_logits(task)
        # set permute_ids, train_ids, all_prompt_labels
        self.load_meta(task)

    def load_target_model_logits(self, task):
        merged_fn = os.path.join(self.ckpt_dir, f'merged-{task}.pt')
        if os.path.exists(merged_fn):
            logits = torch.load(merged_fn)
        else:
            fns = glob(os.path.join(self.ckpt_dir, f'{task}-*.pt'))
            fns = sorted(fns)

            assert len(fns) % self.n_perm == 0
            n_seg = len(fns) // self.n_perm
            # TODO remove hard code to support n_perm != 2
            assert self.n_perm == 2
            print([os.path.basename(f) for f in fns[:n_seg]])
            print([os.path.basename(f) for f in fns[n_seg:]])

            perm1_tensor = torch.cat([torch.load(fn) for fn in fns[:n_seg]])
            perm2_tensor = torch.cat([torch.load(fn) for fn in fns[n_seg:]])
            assert perm1_tensor.shape == perm2_tensor.shape
            logits = torch.stack((perm1_tensor, perm2_tensor)) # [2, n_train_subsets, n_dev, n_class]
            torch.save(logits, merged_fn)

        if self.n_train_sets is not None and self.n_train_sets < logits.shape[1]:
            self.logits = logits[:, :self.n_train_sets]
        else:
            self.logits = logits
            self.n_train_sets = self.logits.shape[1]
        self.n_labels = self.logits.shape[-1]
        print(f"{self.logits.shape}; n_labels: {self.n_labels}; n_train_subsets: {self.n_train_sets}")

    def load_meta(self, task):
        permute_ids = np.load(os.path.join(self.ckpt_dir, f"{task}-permute_ids.npy"))
        train_ids = np.load(os.path.join(self.ckpt_dir, f"{task}-train_ids.npy"))
        all_prompt_labels = get_prompt_labels(task, self.ckpt_dir, train_ids, self.is_unlabel)
        assert len(permute_ids) == len(train_ids) == len(all_prompt_labels)

        self.permute_ids = permute_ids[:self.n_train_sets]
        self.train_ids = train_ids[:self.n_train_sets]
        self.all_prompt_labels = all_prompt_labels[:self.n_train_sets]


def recombine(common_ids, train_labels, n_labels, K, verbose=True):
  valid_ids = []
  for perm_ids in itertools.permutations(common_ids, K):     # P(N, K)
      perm_ids = np.array(list(perm_ids))
      if len(np.unique(train_labels[perm_ids])) < n_labels: continue # exclude [0,0,0,0] or [1,1,1,1]
      valid_ids.append(perm_ids)

  assert len(valid_ids) > 0, "no valid subsets!"
  valid_ids = np.stack(valid_ids)

  if verbose:
      print('# common_ids:', len(common_ids))
      print(common_ids)
      print("valid_ids:", valid_ids.shape)

  return valid_ids


def truncate(selc_ids, n_truncate, useful_size):
    # too many permutations; randomly truncate
    selc_ids = selc_ids.copy()
    np.random.seed(0)
    np.random.shuffle(selc_ids)
    new_ids = selc_ids[:n_truncate]
    assert len(new_ids) == n_truncate

    trunc_useful_size = len(set(new_ids.reshape(-1)))
    if trunc_useful_size != useful_size:
        print(f'WARNING: useful size: {useful_size}, after truncate: {trunc_useful_size}')

    return new_ids


def get_balanced_topN_ids(sorted_ids, train_labels, N, n_labels, min_per_cls=None):
    if min_per_cls is None: min_per_cls = (N//n_labels)
    buckets = np.zeros(n_labels, dtype=int)
    buckets.fill(min_per_cls)

    topN_ids, spare_ids = [], []
    for ex_i in sorted_ids:
        lb = train_labels[ex_i]
        if buckets[lb] > 0:
            topN_ids.append(ex_i)
            buckets[lb] -= 1
        else: # queue
            spare_ids.append(ex_i)

        if buckets.sum() == 0 and len(topN_ids)+len(spare_ids)>=N:
            break

    n_spare = N-len(topN_ids)
    topN_ids.extend(spare_ids[:n_spare])
    topN_ids = np.sort(topN_ids) # make the order stable

    assert len(np.unique(train_labels[topN_ids])) == n_labels
    assert len(topN_ids) == N

    return topN_ids


def flatten_permutation_dimensions(train_ids, permute_ids, n_perm):
    train_ids = np.concatenate([train_ids for _ in range(n_perm)])           # duplicate train_ids
    permute_ids = np.concatenate([permute_ids[:, j] for j in range(n_perm)]) # flatten permutation ids
    assert len(train_ids) == len(permute_ids)
    return train_ids, permute_ids


def dump_selected_subsets(task, model, sel_subset_ids, train_data, method):
    good_train_set = []
    for ids in sel_subset_ids:
        sel_train_data = [train_data[i] for i in ids]

        good_train_set.append(sel_train_data)

    pkl.dump( good_train_set, open(os.path.join(OUT_SELECT, f"{model}-{task}-{method}.pkl"), "wb" ) )


def save_train_ids(task, model, common_ids, sel_subset_ids, method):
    np.save(os.path.join(OUT_SELECT, f'{model}-{task}_common_ids-{method}.npy'), common_ids)
    np.save(os.path.join(OUT_SELECT, f'{model}-{task}_subset_ids-{method}.npy'), sel_subset_ids)

