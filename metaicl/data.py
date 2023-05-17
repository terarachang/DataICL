import os
import csv
import json
import string
import numpy as np
import pickle as pkl
import math
import torch
import pdb

from collections import defaultdict
from functools import partial
from multiprocessing import Pool

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

class MetaICLData(object):

    def __init__(self, logger=None, tokenizer=None, trunc_method="right", use_demonstrations=True, k=16,
                 max_length=1024, max_length_per_example=256, is_opt_model=False,
                 do_tensorize=False, tensorize_dir=None, n_process=None, n_gpu=None, local_rank=-1):

        self.logger = logger
        self.tokenizer = tokenizer
        self.trunc_method = trunc_method
        self.use_demonstrations = use_demonstrations
        self.k = k
        self.max_length = max_length
        self.max_length_per_example = max_length_per_example
        self.is_opt_model= is_opt_model

        self.do_tensorize = do_tensorize
        self.tensorize_dir = tensorize_dir
        self.n_process = n_process
        self.n_gpu = n_gpu
        self.local_rank = local_rank

        self.tensorized_inputs = None
        self.metadata = None

    def __len__(self):
        if self.tensorized_inputs is None:
            return 0
        return len(self.tensorized_inputs["input_ids"])

    def __str__(self):
        text = "[MetaICL Data]: method=%d, "
        if self.use_demonstrations:
            text += "%d demonstrations\n" % self.k
        else:
            text += "no demonstrations\n"
        if self.metadata is None:
            text += "Currently not containing any examples"
        else:
            text += "Currently containing %d examples with %d tensors to be fed in\n" % (len(self.metadata), len(self))
            text += "\n"
            text += self.print_tensorized_example(return_string=True)
        return ("="*50) + "\n" + text + "\n" + ("="*50)

    def get_dataloader(self, batch_size):
        inputs = self.tensorized_inputs
        for k, v in inputs.items():
            if type(v)==list:
                inputs[k] = torch.LongTensor(v)
        shape = inputs["input_ids"].shape
        for v in inputs.values():
            assert v.shape==shape

        n_class = len(self.metadata[0]['options'])
        uni_ids = torch.arange(0, shape[0], n_class)

        inputs["token_type_ids"] = torch.nonzero(inputs["token_type_ids"][uni_ids], as_tuple=True)[1] # [n_data]
        assert len(inputs["token_type_ids"]) == len(uni_ids), len(inputs["token_type_ids"]) - len(uni_ids)
        label_ids = []
        for i, idx in enumerate(inputs["token_type_ids"]):
            label_ids.append(inputs["input_ids"][i*n_class: (i+1)*n_class][:, idx]) # [n_class]
        inputs["labels"] = torch.stack(label_ids) #[n_data, n_class]

        inputs["input_ids"] = inputs["input_ids"][uni_ids]
        inputs["attention_mask"] = inputs["attention_mask"][uni_ids]

        self.logger.info(inputs["input_ids"].shape)

        dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"], inputs["labels"])
        sampler=SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        return dataloader

    def evaluate(self, predictions, groundtruths, is_classification):
        assert len(predictions)==len(self.metadata)
        accs = []
        precisions = defaultdict(list)
        recalls = defaultdict(list)
        for prediction, groundtruth in zip(predictions, groundtruths):
            prediction = prediction.strip()
            groundtruth = [gt.strip() for gt in groundtruth] if type(groundtruth)==list else groundtruth.strip()
            is_correct = prediction in groundtruth if type(groundtruth)==list else prediction==groundtruth
            accs.append(is_correct)

        return np.mean(accs)

    def _truncate(self, inputs, n_keep, method='right'):
        n_truncate = len(inputs) - n_keep

        if method == 'right':
            inputs = inputs[:n_keep]
        elif method == 'left':
            inputs = inputs[n_keep:]
        else: # middle truncate
            mid_i = len(inputs) // 2
            left_i = mid_i - n_truncate // 2
            right_i = left_i + n_truncate
            inputs = inputs[:left_i] + inputs[right_i:]

        assert len(inputs) == n_keep
        return inputs
        


    def _prepro_each_datapoint(self, dp, is_first=True, for_demonstrations=False):
        dp = dp.copy()
        if not is_first:
            dp["input"] = "\n" + dp["input"]

        dp["output"] = " " + dp["output"]
        if "options" in dp:
            dp["options"] = [" " + opt for opt in dp["options"]]

        input_tokens = self.tokenizer(dp["input"])["input_ids"]
        if self.is_opt_model and not is_first: # only keep bos in the first sent
            input_tokens = input_tokens[1:]

        if for_demonstrations:
            output_tokens = self.tokenizer(dp["output"])["input_ids"]
            if self.is_opt_model: output_tokens = output_tokens[1:]

            n_keep = self.max_length_per_example - 2 - len(output_tokens)
            if len(input_tokens) > n_keep:
                input_tokens = self._truncate(input_tokens, n_keep, self.trunc_method)

            return input_tokens, output_tokens

        else:
            assert len(dp["options"])>=2, dp
            assert dp["output"] in dp["options"]

            if self.is_opt_model: # strip bos
                option_tokens = [self.tokenizer(option)["input_ids"][1:] for option in dp["options"]]
            else: # other models do not have a bos token
                option_tokens = [self.tokenizer(option)["input_ids"] for option in dp["options"]]
            option_length = np.max([len(option) for option in option_tokens])

            n_keep = self.max_length_per_example - 2 - option_length
            if len(input_tokens) > n_keep:
                input_tokens = self._truncate(input_tokens, n_keep, self.trunc_method)

            input_tokens = [input_tokens for _ in option_tokens] # expand to (N_options, input_tokens)
            output_tokens = option_tokens                        # tokenized [' negative', ' positive']
            option_tokens = [dp["options"].index(dp["output"])]  #[0] or [1] for binary tasks

            return input_tokens, output_tokens, option_tokens


    def tensorize(self, _train_data, _test_data, options=None):

        if options is not None:
            assert np.all([dp["output"] in options for dp in _train_data])
            for i, dp in enumerate(_test_data):
                assert "options" not in dp
                assert type(dp)==str
                _test_data[i] = {"input": dp, "options": options}

        train_data, test_data = [], []
        if self.use_demonstrations:
            for dp in _train_data:
                assert type(dp)==dict, ("Each example should be a dictionary", dp)
                assert "input" in dp and "output" in dp, ("Training example should contain input and output", dp)
                train_data.append(dp.copy())

        for dp in _test_data:
            assert type(dp)==dict, ("Each example should be a dictionary", dp)
            assert "input" in dp and "options" in dp and type(dp["options"])==list, \
                ("Test example should contain input and options in a list format", dp)
            if "output" not in dp:
                dp["output"] = dp["options"][0] # randomly choose one (we don't need it anyways)
            test_data.append(dp.copy())

        # each datapoint: passage, question, options, output
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        input_ids, attention_mask, token_type_ids = [], [], []
        metadata = []

        if self.use_demonstrations:
            assert len(train_data)==self.k
            demonstrations = []
            for i, dp in enumerate(train_data):
                input_, output_ = self._prepro_each_datapoint(
                    dp, is_first=(i==0), for_demonstrations=True,
                )
                demonstrations += input_ + output_

        for dp_idx, dp in enumerate(test_data):
            inputs, outputs, answer = self._prepro_each_datapoint(
                dp, is_first=not self.use_demonstrations, for_demonstrations=False,
                )

            indices = [[i] for i in range(len(input_ids), len(input_ids)+len(inputs))]

            metadata.append({"indices": indices, "answer": answer, "options": dp["options"]})

            for inputs_, outputs_ in zip(inputs, outputs):
                if self.use_demonstrations:
                    inputs_ = demonstrations + inputs_

                encoded = prepro_sentence_pair_single(
                    inputs_, outputs_, self.max_length, bos_token_id, eos_token_id,
                    allow_truncation=self.use_demonstrations)

                input_ids.append(encoded[0])
                attention_mask.append(encoded[1])
                token_type_ids.append(encoded[2])

        self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids),
                                      attention_mask=torch.LongTensor(attention_mask),
                                      token_type_ids=torch.LongTensor(token_type_ids))
        self.metadata = metadata


    def print_tensorized_example(self, return_string=False):
        assert self.tensorized_inputs is not None

        idx = 0
        text = "Checking the first example..."
        input_ids = self.tensorized_inputs["input_ids"][idx]
        token_type_ids = self.tensorized_inputs["token_type_ids"][idx]
        if type(input_ids)!=list:
            input_ids = input_ids.numpy().tolist()
        if type(token_type_ids)!=list:
            token_type_ids = token_type_ids.numpy().tolist()

        text1 = self.tokenizer.decode(input_ids[:token_type_ids.index(1)])
        text2 = self.tokenizer.decode([_id for _id, _type_id in zip(input_ids, token_type_ids) if _type_id==1])

        if return_string:
            return text

        if self.local_rank<=0:
            self.logger.info("\n"+text1)
            self.logger.info(text2)

def prepro_sentence_pair_single(ids1, ids2, max_length,
                                bos_token_id, eos_token_id,
                                allow_truncation=False):

    if allow_truncation and len(ids1)+len(ids2) > max_length:
        ids1 = ids1[len(ids1)+len(ids2)-max_length:] # left truncate
        assert len(ids1)+len(ids2)==max_length

    n_mask = max_length-len(ids1)-len(ids2)
    assert n_mask>=0, (max_length, len(ids1), len(ids2))
    input_ids = ids1+ids2+[0 for _ in range(n_mask)]
    attention_mask = [1 for _ in ids1+ids2] + [0 for _ in range(n_mask)]
    token_type_ids = [0 for _ in ids1] + [1 for _ in ids2] + [0 for _ in range(n_mask)]
    return input_ids, attention_mask, token_type_ids


