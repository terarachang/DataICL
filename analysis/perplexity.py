import numpy as np
import os
import torch
import pdb
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTJForCausalLM
from utils import get_data, prep_text

class Perplexity():
    def __init__(self, model_id, cache_dir="cached"):
        if "gpt-j" in model_id:
            plm = "EleutherAI/gpt-j-6b"
            self.tokenizer = AutoTokenizer.from_pretrained(plm, cache_dir=cache_dir)
            self.model = GPTJForCausalLM.from_pretrained(plm, cache_dir=cache_dir, 
                    revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
        elif "opt" in model_id:
            plm = f"facebook/{model_id}"
            self.tokenizer = AutoTokenizer.from_pretrained(plm, cache_dir=cache_dir, use_fast=False)
            self.model = AutoModelForCausalLM.from_pretrained(plm, cache_dir=cache_dir, torch_dtype=torch.float16)
        else:
            plm = model_id
            self.tokenizer = AutoTokenizer.from_pretrained(plm, cache_dir=cache_dir)
            self.model = AutoModelForCausalLM.from_pretrained(plm, cache_dir=cache_dir)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()


    # https://huggingface.co/spaces/evaluate-measurement/perplexity/blame/main/perplexity.py
    def compute(self, data, batch_size: int = 32, add_start_token: bool = True):

        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if self.tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(self.tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert (
                len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            self.tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        if add_start_token:
            # leave room for <BOS> token to be added:
            assert (
                self.tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = self.model.config.max_length - 1
        else:
            max_tokenized_len = self.model.config.max_length

        encodings = self.tokenizer(
            data,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]


        # check that each input is long enough:
        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        ppls, likelihoods = [], []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[self.tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(self.device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(self.device), attn_mask], dim=1
                )

            #print(self.tokenizer.decode(encoded_batch[0]))
            labels = encoded_batch

            with torch.no_grad():
                out_logits = self.model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            # CrossEntropyLoss([bs, vocab, len], [bs, len]) * 1[bs, len] # and then mean over len (dim=1)
            nll_batch = (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1) \
                / shift_attention_mask_batch.sum(1)
            perplexity_batch = torch.exp2(nll_batch)

            ppls += perplexity_batch.tolist()
            likelihoods += (-nll_batch).tolist()

        return {"perplexities": np.array(ppls), "log-likelihood": np.array(likelihoods)}


if __name__ == "__main__":
    out_dir = "out_ppl"
    os.makedirs(out_dir, exist_ok=True)

    def save(results, model, task):
        for key, arr in results.items():
            print(f'{model}-{task}-{key}', arr.mean())
            np.save(os.path.join(out_dir, f'{model}-{task}-{key}.npy'), arr)

    for model in ['gpt-j-6b', 'opt-13b']:
        ppl = Perplexity(model)
        for task in ['boolq', 'glue-sst2', 'subj', 'scicite', 'ag_news']:
            train_data, _ = get_data(task)
            proc_data = prep_text(task, train_data)
            print(proc_data[0])

            results = ppl.compute(proc_data, batch_size=100)
            save(results, model, task)

        del ppl 
