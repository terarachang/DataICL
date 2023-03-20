from config._all_tasks_models import *
import pdb
import os
import torch
import numpy as np
from glob import glob

def remove_unused_ids(task, subdir, mode):
    train_ids_fn = os.path.join(subdir, f'{task}-train_ids.npy')
    permute_ids_fn = os.path.join(subdir, f'{task}-permute_ids.npy')
    ori_train_ids = np.load(train_ids_fn)
    ori_perm_ids = np.load(permute_ids_fn)

    logit_shape = torch.load(os.path.join(subdir, f'merged-{task}.pt')).shape

    n_comb = logit_shape[1]
    '''
    print(mode)
    print(logit_shape)
    print(ori_train_ids.shape, '->', ori_train_ids[:n_comb, :].shape)
    print(ori_perm_ids.shape, '->', ori_perm_ids[:n_comb, :2].shape)
    '''
    assert n_comb*2 == logit_shape[0]*logit_shape[1]

    np.save(train_ids_fn, ori_train_ids[:n_comb, :])
    np.save(permute_ids_fn, ori_perm_ids[:n_comb, :2])


def sanity_check(task, subdir, mode):
    assert os.path.exists(os.path.join(subdir, f'{task}-permute_ids.npy'))
    assert os.path.exists(os.path.join(subdir, f'{task}-sampled.pkl'))
    assert os.path.exists(os.path.join(subdir, f'{task}-train_ids.npy'))
    assert os.path.exists(os.path.join(subdir, f'merged-{task}.pt'))

    print(mode)
    logit_shape = torch.load(os.path.join(subdir, f'merged-{task}.pt')).shape
    ids_shape = np.load(os.path.join(subdir, f'{task}-train_ids.npy')).shape
    permute_shape = np.load(os.path.join(subdir, f'{task}-permute_ids.npy')).shape
    print(logit_shape)
    print(ids_shape)
    print(permute_shape)
    


for model in ['opt-6.7b', 'gpt-neo-2.7B']:
    for task in ['glue-sst2', 'ag_news']:
        for mode in ['label']:
            subdir = os.path.join('Dicl', model, f'{mode}_{task}') 
            print(subdir)

            #remove_unused_ids(task, subdir, mode)
            sanity_check(task, subdir, mode)
        print('='*50)

