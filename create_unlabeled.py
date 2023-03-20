import json
import numpy as np
import os
import sys

assert len(sys.argv) == 2, "glue-sst2"
task = sys.argv[1]
fn = f"data/{task}/{task}_500_0_train.jsonl"

is_groundtruth = []
with open(fn, 'r') as f:
    data = []
    for line in f:
        dp = json.loads(line) # dict
        gt = dp['output']
        for opt in dp['options']:
            new_dp = dp.copy()
            new_dp['output'] = opt
            data.append(new_dp)
            is_groundtruth.append(opt == gt)

out_dir = os.path.join("data", task, "unlabeled")
os.makedirs(out_dir, exist_ok="True")
np.save(os.path.join(out_dir, 'is_groundtruth.npy'), is_groundtruth)

with open(os.path.join(out_dir, f"{task}_500_0_train.jsonl"), "w") as fout:
    for line in data:
        fout.write(json.dumps(line)+"\n")

