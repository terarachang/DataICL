import numpy as np
import os
import pdb
import torch
import torch.nn.functional as F

from tqdm import tqdm
from transformers import Adafactor, AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForCausalLM, GPTJForCausalLM, GPTNeoForCausalLM


class MetaICLModel(object):

    def __init__(self, logger=None, out_dir=None, fp16=True, local_rank=-1):
        if logger is None:
            class Logger():
                def info(self, text):
                    print ("Logging from MetaICLModel:\t", text)
            logger = Logger()

        self.logger = logger
        self.out_dir = out_dir
        self.fp16 = fp16
        self.local_rank = local_rank

        if self.local_rank == -1:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            n_gpu = torch.cuda.device_count()
            ws = 1
        else:  # distributed mode
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            ws = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
            torch.distributed.init_process_group(backend="nccl")
            n_gpu = 1

        self.n_gpu = n_gpu
        self.device = device
        if self.local_rank <= 0:
            logger.info("Setting up for local_rank=%d, world_size=%d" % (self.local_rank, ws))
        self.model_name = None
        self.model = None
        self.mode = None

    def __str__(self):
        text = "[MetaICL Model]: "
        if self.model_name is None:
            text += "No model loaded yet"
        else:
            text += self.model_name
            if self.mode is None:
                text += " (no mode setted - try .train() or .eval()"
            else:
                text += " (%s mode)" % self.mode
        text += "\nusing device %s, %d gpus, local_rank=%d" % (self.device, self.n_gpu, self.local_rank)
        return ("="*50) + "\n" + text + "\n" + ("="*50)

    def is_none(self):
        return self.model is None

    def train(self):
        self.model.train()
        self.mode = "train"

    def eval(self):
        self.model.eval()
        self.mode = "eval"

    def cuda(self):
        self.model.cuda()

    def to_device(self):
        self.model.to(self.device)

    def load(self, checkpoint=None, gpt2="gpt2-large"):
        '''
        checkpoint can be either keyword of the model or path to the checkpoint file
        '''
        if checkpoint is None:
            self.logger.info("Loading the model...")
            if gpt2.startswith("gpt2"):
                model = AutoModelForCausalLM.from_pretrained(gpt2, cache_dir="cached") 
            elif "opt" in gpt2:
                model = AutoModelForCausalLM.from_pretrained(f"facebook/{gpt2}", cache_dir="cached",
                    torch_dtype=torch.float16)
            elif "bloom" in gpt2:
                model = AutoModelForCausalLM.from_pretrained(f"bigscience/{gpt2}", cache_dir="cached",
                    device_map='auto', load_in_8bit=True)
            elif "gpt-j" in gpt2:
                model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6b", cache_dir="cached", 
                    revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
            elif "gpt-neo-" in gpt2:
                model = GPTNeoForCausalLM.from_pretrained(f"EleutherAI/{gpt2}", cache_dir="cached")
            else:
                raise NotImplementedError(checkpoint)
            self.model_name = gpt2
        else:
            self.model_name = checkpoint

            assert os.path.exists(checkpoint), checkpoint
            if self.local_rank <= 0:
                self.logger.info("Loading the model from %s" % checkpoint)
            state_dict = torch.load(checkpoint)
            model = AutoModelForCausalLM.from_pretrained(gpt2, state_dict=state_dict)

        self.model = model

    def save(self, step):
        if self.local_rank <= 0:
            model_state_dict = {key[7:] if key.startswith("module.") else key: value.cpu()
                                for key, value in self.model.state_dict().items()}
            torch.save(model_state_dict, os.path.join(self.out_dir, "model-{}.pt".format(step)))
            self.logger.info("Saving model parameters at step=%d" % step)

    def parallel(self):
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        if self.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank)


    def do_inference(self, data, batch_size=1, verbose=False):
        dataloader = data.get_dataloader(batch_size)
        if verbose:
            dataloader = tqdm(dataloader)
        offset = 0
        n_class = len(data.metadata[0]['options'])
        probs = torch.FloatTensor(len(data)*n_class)
        for batch in dataloader:
            input_ids=batch[0].to(self.device)
            attention_mask=batch[1].to(self.device)
            token_type_ids=batch[2].to(self.device)
            if len(batch)==3:
                labels=None
            else:
                labels=batch[3].to(self.device)

            with torch.no_grad():
                prob = self.run_model(input_ids, attention_mask, token_type_ids, labels=labels)
            bs = len(input_ids) * n_class
            probs[offset: offset+bs] = prob.cpu()
            offset += bs

        assert offset == len(probs)
        return probs

    def do_predict(self, data, batch_size=1, probs=None, verbose=False):
        predictions, label_probs = [], []
        for idx, dp in enumerate(data.metadata):
            curr_label_prob = torch.FloatTensor([(probs[indices]).sum() for indices in dp["indices"]])
            prediction_idx = sorted(enumerate(curr_label_prob), key=lambda x: x[1], reverse=True)[0][0] #idx of the highest prob 
            prediction = dp["options"][prediction_idx]
            predictions.append(prediction.strip())
            label_probs.append(curr_label_prob)
        label_probs = torch.stack(label_probs)
        return predictions, label_probs

    def run_model(self, input_ids, attention_mask, label_position, labels=None):
        # label_position: [bs], labels: [bs, n_class]
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # shift
        logits = outputs.logits[..., :-1, :].contiguous() #[bs, length, vocab_size]
        label_position = label_position -1 # [bs]
        bs = len(logits)
        _logits = logits[range(bs), label_position]  # [bs, vocab_size] 
        prob = _logits[:, labels[0]].view(-1)        # flatten [bs, n_class]

        return prob



