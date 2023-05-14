import os
import itertools
import argparse
import pickle as pkl
import random
import torch
import torch.nn as nn
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
from config.config import OUT_SELECT

def apply_template(text, task):
    if task == 'glue-sst2':
        return f"Review: {text}\nSentiment:"
    elif task == 'ag_news':
        return f"Article: {text}\nAnswer:"
    elif task == 'boolq':
        return f"Exercise: read the text and answer the question by yes or no.\n\nText: {text}\nQuestion: ?"
    elif task == 'subj':
        return f"Input: {text}\nType:"
    elif task == 'scicite':
        return f'Is the following citation from a scientific paper describing a method, a result, or background?\n"{text}"\nAnswer:'

def create_content_free_data(copy_dp, cf_inputs):
    task = copy_dp['task']
    cf_data = []
    for inp in cf_inputs:
        cf_dp = {'task': task,
          'input': apply_template(inp, task),
          'options': copy_dp['options']
        }
        cf_data.append(cf_dp)

    return cf_data


def main(logger, args):

    if args.gpt2.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2, cache_dir="cached")
    elif "gpt-j" in args.gpt2:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b", cache_dir="cached")
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
    eval_data = load_data(args.split, 500, seed, args.dataset)

    n_class = len(eval_data[0]["options"])
    test_task = eval_data[0]["task"]
    logger.info("-"*50)
    logger.info(f"Seed: {seed}, Task: {test_task}, # Class: {n_class}")
    logger.info(f"[{args.split}]: {len(eval_data)}")

    content_free_data = create_content_free_data(eval_data[0], cf_inputs=["N/A", "", "[MASK]"])

    logger.info(f"Eval {args.mode}")
    # load prompts of different subset selection methods
    test_task_ = args.source_task if args.source_task is not None else test_task
    selected_train_fn = os.path.join(OUT_SELECT, f"{args.gpt2.lower()}-{test_task_}-{args.mode}.pkl")
    selected_train_data = pkl.load( open(selected_train_fn, "rb" ) )
    n_prompts = len(selected_train_data)

    logger.info(f"# prompts: {n_prompts}")
    all_probs_bf = torch.zeros((n_prompts, len(eval_data), n_class))
    all_probs_calib = torch.zeros((n_prompts, len(eval_data), n_class))
    all_perf = np.zeros((n_prompts))

    for ti, curr_train_data in enumerate(selected_train_data):
        logger.info(f"Selected Subset-{ti+1}...")
        assert len(curr_train_data) == args.k

        ### calibration
        all_logits = []
        for cf_dp in content_free_data:
            logits = run(logger, test_task, metaicl_data, metaicl_model, 
                                curr_train_data, [cf_dp.copy()], 
                                seed, args.is_classification, n_class)
            all_logits.append(logits)

        all_logits = torch.cat(all_logits, 0)      # [n_cf, n_class]
        all_p_y = nn.Softmax(-1)(all_logits)          # logits -> probs
        p_cf = torch.mean(all_p_y, axis=0)         # avg over context-free inputs

        W_cf = torch.eye(n_class)*1/p_cf
        ### calibration

        all_probs_bf[ti], all_probs_calib[ti], all_perf[ti] = \
                run(logger, test_task, metaicl_data, metaicl_model, 
                    curr_train_data, eval_data, 
                    seed, args.is_classification, n_class, W_cf)

    save_performance(args, test_task, all_probs_bf, all_probs_calib, all_perf)


def save_performance(args, task, probs_bf, probs_calib, all_perf):
    cache_dir = os.path.join(args.results_dir, 'calibrated', task, args.gpt2)
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(cache_dir)
    torch.save(probs_bf, os.path.join(cache_dir, f'{args.mode}-{args.split}-probs_before.pt'))
    torch.save(probs_calib, os.path.join(cache_dir, f'{args.mode}-{args.split}-probs_calib.pt'))
    np.save(os.path.join(cache_dir, f'{args.mode}-{args.split}-acc.npy'), all_perf)
    print('Acc:')
    print(repr(all_perf))


def run(logger, task, metaicl_data, metaicl_model, train_data, eval_data, seed,
        is_classification, n_class, W_cf=None):

    has_gt = 'output' in eval_data[0]
    metaicl_data.tensorize(train_data, eval_data)
    metaicl_data.print_tensorized_example()

    logits = metaicl_model.do_inference(metaicl_data, args.test_batch_size, verbose=False)
    if W_cf is not None:
        probs_before = nn.Softmax(-1)(logits.view(-1, n_class))
        logits = torch.matmul(probs_before, W_cf.T)
        probs_calib = nn.Softmax(-1)(logits)
        logits = logits.view(-1)

    predictions, logits = metaicl_model.do_predict(metaicl_data, probs=logits)
    if has_gt:
        groundtruths = [dp["output"] for dp in eval_data]
        perf = metaicl_data.evaluate(predictions, groundtruths, args.is_classification)
        logger.info("Accuracy=%s" % perf)
        return probs_before, probs_calib, perf
    else:
        return logits 




if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_demonstrations", default=False, action="store_true")
    parser.add_argument('--is_classification', action='store_false')
    parser.add_argument("--trunc_method", type=str, default='right', choices=['right', 'left', 'middle'])
    parser.add_argument("--dataset", type=str, default=None, required=True)
    parser.add_argument("--source_task", type=str, default=None, help="prompts from a different task; OOD experiments")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--max_length_per_example", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default="final_results")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--gpt2", type=str, default="gpt-j-6b")
    parser.add_argument("--mode", type=str, default="Random")
    parser.add_argument("--log_file", default=None, type=str)

    args = parser.parse_args()

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
