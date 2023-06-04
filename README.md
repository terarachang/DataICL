## DataICL

![Python](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/pytorch-1.12-green.svg?style=plastic)
![transformers](https://img.shields.io/badge/transformers-4.20.1-green.svg?style=plastic)
![GPU](https://img.shields.io/badge/RTX-A6000-green.svg?style=plastic)


> **Data Curation Alone Can Stabilize In-context Learning (ACL 2023)**<br>
> Ting-Yun Chang and Robin Jia<br>

> :film_strip: https://www.youtube.com/watch?v=ZVz5pI06FRE

> :scroll: https://arxiv.org/pdf/2212.10378.pdf

>
> **Abstract:** *In-context learning (ICL) enables large language models (LLMs) to perform new tasks by prompting them with a sequence of training examples. However, it is known that ICL is very sensitive to the choice of training examples: randomly sampling examples from a training set leads to high variance in performance. In this paper, we show that carefully curating a subset of training data greatly stabilizes ICL performance without any other changes to the ICL algorithm (e.g., prompt retrieval or calibration). We introduce two methods to choose training subsets -- both score training examples individually, then select the highest-scoring ones. CondAcc scores a training example by its average dev-set ICL accuracy when combined with random training examples, while Datamodels learns linear regressors that estimate how the presence of each training example influences LLM outputs. Across five tasks and two LLMs, sampling from stable subsets selected by CondAcc and Datamodels improves average accuracy over sampling from the entire training set by 7.7% and 6.3%, respectively. Surprisingly, the stable subset examples are not especially diverse in content or low in perplexity, in contrast with other work suggesting that diversity and perplexity are important when prompting LLMs.*

> This repository is based on [MetaICL](https://github.com/facebookresearch/MetaICL#metaicl-learning-to-learn-in-context)

## Content

1. [Quick Start](#quick-start)
2. [CondAcc](#condacc)
3. [Datamodels](#datamodels)
4. [Baselines](#baselines)
5. [Evaluation](#evaluation)
    - [Labeled](#labeled-setup)
    - [Unlabeled](#unlabeled-setup)
    - [OOD](#ood-setup)
6. [Data](#data)
7. Construct $\mathcal{D}_{\text{ICL}}$
8. How to use the released $\mathcal{D}_{\text{ICL}}$?
9. [Stable Subset Examples](#stable-subset-examples)

## Quick Start
- ```$ pip install -r requirements.txt```
- ```$ bash demo/download_dicl.sh``` will download the released prompt-output pairs
    - see the Construct $\mathcal{D}_{\text{ICL}}$ section below for more details

## CondAcc
- The proposed CondAcc method is implemented in [`select_condacc.py`](select_condacc.py)
- To reproduce the results in the paper, see [Evaluation](#evaluation)

## Datamodels
- To train datamodels, run:
```bash
$ bash scripts/train_datamodels.sh
```
- To test datamodels (Appendix A3), run:
```bash
$ bash scripts/test_datamodels.sh
```
- The Datamodels selection is implemented in [`select_datamodels.py`](select_datamodels.py)
- To reproduce the results in the paper, see [Evaluation](#evaluation)
- Download pretrained datamodels [`out_datamodel.zip`](https://drive.google.com/file/d/1Z9Fci7bOU9WLvgFI_0y2iTTJgiFKfYhM/view?usp=sharing)

## Baselines
- The **Oneot** baseline:
    - First, run 1-shot ICL by `$ bash scripts/run_oneshot.sh {gpu_i}`
    - The subset selection process is implemented in [`baseline_oneshot.py`](baseline_oneshot.py)
- The **TopPrompts** baseline is implemented in [`baseline_top_prompts.py`](baseline_top_prompts.py)
- The **Calib** baseline is implemented in [`calib_evaluate.py`](calib_evaluate.py)
    - `$ bash scripts/run_calibration.sh {gpu_i}`
- To reproduce the results in the paper, see [Evaluation](#evaluation)


## Evaluation
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

## Data
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


## Construct $\mathcal{D}_{\text{ICL}}$
- We release the set of prompt-output pairs $\mathcal{D}_{\text{ICL}}$ in [`Dicl`](https://drive.google.com/file/d/1gKueGgRjVKWZ5RXE9PBVyCD5dvbLYdRk/view?usp=sharing). The files are organized as follow:

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
- Each task has three folders, e.g. `label_glue-sst2` and `unlabel_glue-sst2` for labeled/unlabeled setup (Sec 2), and `test_glue-sst2` for evaluating how well Datamodels can approximate the target LLM on the held-out prompts (Appendix A3).
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
        - add the argument `--is_unlabel` to build $\mathcal{D}_{\text{ICL}}$ for the unlabeled setup.
    - *Note*: the construction process may take hundreds of GPU hours for a task
  
## How to use the released $\mathcal{D}_{\text{ICL}}$?
- We include an example code in [`demo`](demo/demo_dicl.py)
- ```$ bash demo/download_dicl.sh``` will automatically download and unzip [`Dicl.zip`](https://drive.google.com/file/d/1gKueGgRjVKWZ5RXE9PBVyCD5dvbLYdRk/view?usp=sharing)
- ```$ python -m demo.demo_dicl --model gpt-j-6b --task glue-sst2```, use `--is_unlabel` for the unlabeled setup.
    -  this will return a list of datapoints, where a datapoint is a dict that looks like: 
    - ```python     
      {'train_examples': [{   'input': 'Review: whole mess \nSentiment:',
                              'options': ['negative', 'positive'],
                              'output': 'negative',
                              'task': 'glue-sst2'},
                          {   'input': 'Review: but it also comes with the '
                                       'laziness and arrogance of a thing that '
                                       "already knows it 's won . \n"
                                       'Sentiment:',
                              'options': ['negative', 'positive'],
                              'output': 'negative',
                              'task': 'glue-sst2'},
                          {   'input': 'Review: intelligent and moving . \n'
                                       'Sentiment:',
                              'options': ['negative', 'positive'],
                              'output': 'positive',
                              'task': 'glue-sst2'},
                          {   'input': 'Review: does point the way for '
                                       'adventurous indian filmmakers toward a '
                                       'crossover into nonethnic markets . \n'
                                       'Sentiment:',
                              'options': ['negative', 'positive'],
                              'output': 'positive',
                              'task': 'glue-sst2'}],
        'train_ids': np.array([101, 286, 666, 623]),
        'dev accuracy': 0.85,
        'logits': a torch.FloatTensor of shape [n_dev, n_labels]}
        ```
        
## Stable Subset Examples
- We include the identified stable subset examples in [`out_select`](out_select)
    - `label_stable_subsets/{model}-{task}-{CondAcc/Datamodels}.jsonl` each file shows a stable subset (20 examples) identified by CondAcc/Datamodels in the labeled setup
    - `unlabel_stable_subsets/{model}-{task}-CondAcc.jsonl` each file shows a stable subset (20 examples) identified by CondAcc in the unlabeled setup
    - `good_example_ids/*.npy` each file shows the corresponding 20 example IDs
