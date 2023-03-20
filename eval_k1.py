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
import pprint

from tqdm import tqdm
from collections import Counter, defaultdict

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import GPT2Tokenizer, AutoTokenizer

from metaicl.model import MetaICLModel
from metaicl.data import MetaICLData
from utils.data import load_data, random_subset_of_comb
from config.config import OUT_ONESHOT


def get_data(task, model, n, split='train'):
    fn = f"data/{task}/{task}_{n}_0_{split}.jsonl"
    data = []
    with open(fn, "r") as f:
        for line in f:
            dp = json.loads(line)
            data.append([dp])
    return data


def main(logger, args):

    if args.gpt2.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2, cache_dir="cached")
    elif "gpt-j" in args.gpt2:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir="cached")
    elif "gpt-neo-" in args.gpt2:
        tokenizer = GPT2Tokenizer.from_pretrained(f"EleutherAI/{args.gpt2}", cache_dir="cached")
    elif "opt" in args.gpt2:
        tokenizer = GPT2Tokenizer.from_pretrained(f"facebook/{args.gpt2}", cache_dir="cached")

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

    logger.info("batch_size=%d\tmax_length=%d\tmax_length_per_example=%d" % (
        args.test_batch_size, max_length, max_length_per_example))

    metaicl_data = MetaICLData(logger, tokenizer, args.trunc_method, args.use_demonstrations, args.k,
                               max_length, max_length_per_example, is_opt_model=("opt" in args.gpt2))

    seed = args.seed # shorten name
    dev_data = load_data(args.split, 500, seed, args.dataset)

    n_class = len(dev_data[0]["options"])
    test_task = dev_data[0]["task"]
    logger.info("-"*50)
    logger.info(f"Seed: {seed}, Task: {test_task}, # Class: {n_class}")
    logger.info(f"[Dev]: {len(dev_data)}")

    all_train_data = get_data(test_task, args.gpt2, 500, 'train')
    n_prompts = len(all_train_data)


    n_prompts = len(all_train_data)
    logger.info(f"# prompts: {n_prompts}")
    all_probs = torch.zeros((n_prompts, len(dev_data), n_class))

    for ti, curr_train_data in enumerate(tqdm(all_train_data, total=n_prompts)):
        logger.info(f"Selected Subset-{ti+1}...")
        assert len(curr_train_data) == args.k

        all_probs[ti] = run(logger, test_task, metaicl_data, metaicl_model, 
                                curr_train_data, dev_data, 
                                seed, args.is_classification)
        if ti % 100 == 0:
            save_probs(args, test_task, all_probs)
            print(ti, "saved!")

    save_probs(args, test_task, all_probs)


def save_probs(args, task, probs):
    torch.save(probs, os.path.join(args.out_dir, f'{args.split}-logits.pt'))


def run(logger, task, metaicl_data, metaicl_model, train_data, dev_data, seed,
        is_classification):

    metaicl_data.tensorize(train_data, dev_data)
    metaicl_data.print_tensorized_example()

    probs = metaicl_model.do_inference(metaicl_data, args.test_batch_size, verbose=False)

    predictions, probs = metaicl_model.do_predict(metaicl_data, probs=probs)
#    groundtruths = [dp["output"] for dp in dev_data]
#    perf = metaicl_data.evaluate(predictions, groundtruths, is_classification)
#    logger.info("Accuracy=%s" % perf)

    return probs



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_demonstrations", default=False, action="store_true")
    parser.add_argument('--is_classification', action='store_false')
    parser.add_argument("--trunc_method", type=str, default='right', choices=['right', 'left', 'middle'])
    parser.add_argument("--dataset", type=str, default=None, required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--max_length_per_example", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--gpt2", type=str, default="gpt-j-6b")
    parser.add_argument("--log_file", default=None, type=str)

    args = parser.parse_args()
    args.out_dir = os.path.join(OUT_ONESHOT, args.dataset, args.gpt2)

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
