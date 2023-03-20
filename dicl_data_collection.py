import os
import itertools
import argparse
import pickle as pkl
import random
import torch
import math
import json
import string
import logging
import numpy as np
import pdb
from tqdm import tqdm
from collections import Counter, defaultdict

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import GPT2Tokenizer, AutoTokenizer

from metaicl.model import MetaICLModel
from metaicl.data import MetaICLData

from utils.data import load_data, random_subset_of_comb, balanced_subset_of_comb
from config.config import OUT_DATA_COLLECT
from utils.selection import setup


def main(logger, args):

    if args.gpt2.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2, cache_dir="cached")
    elif "gpt-j" in args.gpt2:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir="cached")
    elif "gpt-neo-" in args.gpt2:
        tokenizer = GPT2Tokenizer.from_pretrained(f"EleutherAI/{args.gpt2}", cache_dir="cached")
    elif "opt" in args.gpt2:
        tokenizer = AutoTokenizer.from_pretrained(f"facebook/{args.gpt2}", cache_dir="cached", use_fast=False)

    ### checkpoint ...
    checkpoint = args.checkpoint
    metaicl_model = MetaICLModel(logger)

    metaicl_model.load(checkpoint, gpt2=args.gpt2)
    metaicl_model.to_device()
    metaicl_model.eval()

    # setup hyperparams for data
    max_length_per_example = args.max_length_per_example
    if args.use_demonstrations:
        max_length = min(max_length_per_example * args.k, 1024)
    else:
        max_length = max_length_per_example

    seed, n_comb, n_perm = args.seed, args.n_comb, args.n_perm # shorten names

    logger.info("batch_size=%d\tmax_length=%d\tmax_length_per_example=%d" % (
        args.test_batch_size, max_length, max_length_per_example))

    metaicl_data = MetaICLData(logger, tokenizer, args.trunc_method, args.use_demonstrations, args.k,
                               max_length, max_length_per_example, is_opt_model=("opt" in args.gpt2))

    train_data = load_data("train", 500, seed, args.dataset, 
        template_dir="unlabeled" if args.is_unlabel else "")
    dev_data = load_data(args.split, 500, seed, args.dataset)

    permute_ids = list(itertools.permutations(list(range(args.k))))
    n_class = len(dev_data[0]["options"])
    test_task = dev_data[0]["task"]
    logger.info("-"*50)
    logger.info(f"Seed: {seed}, Task: {test_task}, # Class: {n_class}")
    logger.info(f"[Dev]: {len(dev_data)}")

    # sample a set of prompts
    # the total number of prompts = n_comb * n_perm
    sampled_train_fn = os.path.join(args.out_dir, f"{test_task}-sampled.pkl")
    try:
        sampled_train_data = pkl.load( open(sampled_train_fn, "rb" ) )
        n_comb = len(sampled_train_data) 
    except FileNotFoundError:
        logger.info("Sample combinations...")

        random.seed(seed)
        if args.n_labels == 2:
            sampled_ids = random_subset_of_comb(range(len(train_data)), args.k, n_comb)
        else:
            train_labels = np.array([dp['options'].index(dp['output']) for dp in train_data])
            sampled_ids = balanced_subset_of_comb(range(len(train_data)), args.k, n_comb, train_labels)

        sampled_train_data = []
        for tr_set in sampled_ids:
            sampled_train_data.append([train_data[i] for i in tr_set])

        assert len(sampled_ids) == n_comb == len(sampled_train_data)

        pkl.dump( sampled_train_data, open(sampled_train_fn, "wb" ) )
        np.save(os.path.join(args.out_dir, f"{test_task}-train_ids.npy"), sampled_ids)

    logger.info(f"Premute id = {args.permute_fn_id}")
    sampled_permute_fn = os.path.join(args.out_dir, f"{test_task}-permute_ids.npy")
    try:
        sampled_permute_i = np.load(sampled_permute_fn)[:,args.permute_fn_id]
    except FileNotFoundError:
        logger.info("Sample permutations...")
        all_permute_i = np.zeros((n_comb, n_perm), dtype='int')
        np.random.seed(seed)
        for i in range(n_comb):
            all_permute_i[i] = np.random.choice(len(permute_ids), n_perm, replace=False)
        np.save(sampled_permute_fn, all_permute_i)
        sampled_permute_i = all_permute_i[:,args.permute_fn_id]

    # support running different segments on different gpus at the same time
    assert len(sampled_train_data) == len(sampled_permute_i)
    assert args.segment_id < args.n_segments
    assert args.n_comb % args.n_segments == 0
    n_comb = n_comb // args.n_segments 
    logger.info('-'*50)
    logger.info(f'[{args.segment_id*n_comb}, {(args.segment_id+1)*n_comb}]')
    sampled_permute_i = sampled_permute_i[args.segment_id*n_comb: (args.segment_id+1)*n_comb]
    sampled_train_data = sampled_train_data[args.segment_id*n_comb: (args.segment_id+1)*n_comb]

    # run ICL with different prompts
    all_probs = torch.zeros((n_comb, len(dev_data), n_class))
    for ti, (permute_i, curr_train_data) in enumerate(tqdm(zip(sampled_permute_i, sampled_train_data), total=n_comb)):
        assert len(curr_train_data) == args.k
        permuted_train_data = [curr_train_data[i] for i in permute_ids[permute_i]]

        all_probs[ti] = run(logger, test_task, metaicl_data, metaicl_model, 
                                permute_i, permuted_train_data, dev_data, 
                                seed, args.is_classification)

        if ti % 100 == 0:
            save_probs(args, test_task, all_probs)
            print(ti, "saved!")

    save_probs(args, test_task, all_probs)


def run(logger, task, metaicl_data, metaicl_model, permute_i, train_data, dev_data, seed,
        is_classification):

    metaicl_data.tensorize(train_data, dev_data)
    metaicl_data.print_tensorized_example()


    probs = metaicl_model.do_inference(metaicl_data, args.test_batch_size, verbose=False)

    predictions, probs = metaicl_model.do_predict(metaicl_data, probs=probs)
    #groundtruths = [dp["output"] for dp in dev_data]
    #perf = metaicl_data.evaluate(predictions, groundtruths, is_classification)
    #logger.info("Accuracy=%s" % perf)

    return probs 

def save_probs(args, task, probs):
    cache_path = os.path.join(args.out_dir, f"{task}-k={args.k}-p={args.permute_fn_id}-s={args.segment_id}")
    logger.info(cache_path)
    torch.save(probs, f'{cache_path}.pt')


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--is_unlabel", action="store_true")
    parser.add_argument("--use_demonstrations", default=False, action="store_true")
    parser.add_argument('--is_classification', action='store_false')
    parser.add_argument("--trunc_method", type=str, default='right', choices=['right', 'left', 'middle'])
    parser.add_argument("--dataset", type=str, default=None, required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--max_length_per_example", type=int, default=128)
    parser.add_argument("--permute_fn_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_comb", type=int, default=25000, 
        help="the number of distinct combinations of training examples")
    parser.add_argument("--n_perm", type=int, default=2, 
        help="the number of permutations under the same combination")
    parser.add_argument("--n_segments", type=int, default=5,
        help="divide prompts into different segments for multi-gpu speedup")
    parser.add_argument("--segment_id", type=int, default=0)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--gpt2", type=str, default="gpt-j-6b")
    parser.add_argument("--log_file", default=None, type=str)

    args = parser.parse_args()
    label_dir = 'unlabel' if args.is_unlabel else 'label'
    args.out_dir = os.path.join(OUT_DATA_COLLECT, args.gpt2, args.dataset, label_dir)
    args.n_labels, _ = setup(args.dataset)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    handlers = [logging.StreamHandler()]
    if args.log_file is not None:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logger = logging.getLogger(__name__)
    logger.info(args)

    main(logger, args)
