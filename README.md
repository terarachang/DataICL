## Careful Data Curation Stabilizes In-context Learning

![Python](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/pytorch-1.12-green.svg?style=plastic)
![transformers](https://img.shields.io/badge/transformers-4.20.1-green.svg?style=plastic)
![GPU](https://img.shields.io/badge/RTX-A6000-green.svg?style=plastic)

*This repository is modified from [MetaICL](https://github.com/facebookresearch/MetaICL#metaicl-learning-to-learn-in-context)

### CondAcc
- The proposed CondAcc method is implemented in `select_condacc.py`
- To reproduce the results in the paper, see [Evaluation](#Evaluation)

### Datamodels
- To train datamodels, run:
```bash
$ bash scripts/train_datamodels.sh
```
- The Datamodels selection is implemented in `select_datamodels.py`
- To reproduce the results in the paper, see [Evaluation](#Evaluation)

### Baselines
- The **Oneot** baseline:
    - First, run 1-shot ICL by `$ bash scripts/run_oneshot.sh {gpu_i}`
    - The subset selection process is implemented in `baseline_oneshot.py`
- The **TopPrompts** baseline is implemented in `baseline_top_prompts.py`
- The **Calib** baseline is implemented in `calib_evaluate.py`
    - `$ bash scripts/run_calibration.sh {gpu_i}`
- To reproduce the results in the paper, see [Evaluation](#Evaluation)


### Evaluation
The following scripts will run the two proposed methods and all baselines.
#### Labeled Setup
```bash
$ bash scripts/run_label.sh {gpu_i}
```
#### Unlabeled Setup
```bash
$ bash scripts/run_unlabel.sh {gpu_i}
```

#### OOD Setup
```bash
$ bash scripts/run_ood.sh {gpu_i}
```

### Data
- We include the data in [`data/`](data/). The files are organized as follow:
```
data
├── glue-sst2
│   ├── *train.jsonl
│   ├── *dev.jsonl
│   ├── *test.jsonl
│   └── unlabeled
│       ├── *train.jsonl
│       └── is_groundtruth.npy
├── boolq/
├── subj/
├── scicite/
├── glue-mnli/
└── ag_news/
```
- Each task folder contains `*train.jsonl, *dev.jsonl, *test.jsonl`, the gold-labeled train/dev/test splits, and the unlabeled training set `unlabeled/*train.jsonl`. 
- Note that both the labeled and unlabeled setups use the same dev sets for method developement and are evaluated on the same test sets.
- To reproduce the data creation for the unlabeled setups, see `create_unlabeled.py`


### Construct $\mathcal{D}_{\text{ICL}}$
- We release the set of prompt-output pairs $\mathcal{D}_{\text{ICL}}$ in [`Dicl`](https://drive.google.com/file/d/1_ZhDS__fF49DBydKu3pyoby0OOTEwQA2/view?usp=sharing). The files are organized as follow:

```
Dicl
├── gpt-j-6b
│   ├── label_glue-sst2
│   │   ├── *train_ids.npy
│   │   ├── *permute_ids.npy
│   │   ├── *sampled.pkl
│   │   └── merged*.pt
│   ├── unlabel_glue-sst2/
│   ├── label_boolq/
│   ├── unlabel_boolq/
│   ├── label_subj/
│   ├── unlabel_subj/
│   ├── label_scicite/
│   ├── unlabel_scicite/
│   ├── label_ag_news/
│   └── unlabel_ag_news/
├── opt-13b/
├── opt-6.7b/
└── gpt-neo-2.7B/

```
- Each task folder contains 4 files:
    - `*sampled.pkl`: the list of sampled prompts, where each prompt consists of a list of $K$ training examples.
    - `*train_ids.npy`: the training example IDs in each prompt, where each row in the array consists of $K$ example IDs.
    - `*permute_ids.npy`: the permutation IDs of each prompt.
    - `merged*.pt`: given the prompts, the LLM's output logits before softmax.

- To reproduce the construction of $\mathcal{D}_{\text{ICL}}$:
    - ```$ bash scripts/build_dicl.sh {gpu_i} {segment_i} {permute_i}```
        - `gpu_i`: the available gpu id, default 0.
        - `segment_i`: [0-4]. We divide the list of sampled prompts into 5 segments to run them in parallel.
        - `permute_i`: [0-1]. Given the same prompt, we run 2 different permutations.
        - add the argument `--is_unlabel` to build $\mathcal{D}_{\text{ICL}}$ for the unlabeled setup
    - *Note*: the construction process may take hundreds of GPU hours for a task


