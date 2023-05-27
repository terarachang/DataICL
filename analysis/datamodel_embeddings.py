import os
import numpy as np
import torch
import argparse
from config.config import *
from utils.selection import *


def get_datamodel_embeds(args):
    # datamodels are linear; thus, we can use their weights as meaningful embeddings

    dirs = sorted(glob(f"{OUT_DM}/{args.model}/{args.task}/*00"))
    path = dirs[-1]
    print(path, [os.path.basename(d) for d in dirs])
    n_train = 999 if args.task in ["glue-mnli", "scicite"] else 1000
    
    # we have n_dev datamodels in total, each has (n_shots*n_train) weights
    weights = torch.load(os.path.join(path, 'weights.pt'))[:,:-1].numpy() # the -1 dimension is the bias term
    n_dev = weights.shape[0]
    weights = weights.reshape(n_dev, n_train, args.n_shots)
    print(weights.shape)
  
    embeddings = weights.transpose((1,0,2)).reshape(n_train, -1) # (n_train, n_dev*n_shots)
    print(embeddings.shape)
    return embeddings


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-j-6b", required=True)
    parser.add_argument("--task", type=str, default="glue-sst2", required=True)
    args = parser.parse_args()
    print(args)

    args.n_labels, args.n_shots = setup(args.task)
    embeddings = get_datamodel_embeds(args)
