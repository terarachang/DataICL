from transformers import AutoTokenizer, AutoModel
import argparse
import torch
import torch.nn.functional as F
import json
import pdb
import sys
import os
from utils import get_data, prep_text
sys.path.append(os.path.dirname(os.path.abspath('config')))
from config.config import OUT_SENT_EMBED



def mean_pooling(model_output, attention_mask):
    # Mean Pooling - Take attention mask into account for correct averaging
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_sentbert_embed(model, encoded_input):
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.cpu()

def get_roberta_embed(model, encoded_input):
    with torch.no_grad():
        sentence_embeddings = model(**encoded_input).last_hidden_state[:, 0]

    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.cpu()


def get_split_embeds(model, tokenizer, task, split, plm, embed_dim, bs, verbose):
    data, _ = get_data(task, split)
    sentences = prep_text(task, data)
    if verbose:
        print(sentences[0], split)
    
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    encoded_input['input_ids'] = encoded_input['input_ids'].cuda()
    encoded_input['attention_mask'] = encoded_input['attention_mask'].cuda()

    sentence_embeddings = torch.zeros((len(sentences), embed_dim))
    for i in range(0, len(sentences), bs):
        batch_input = {
            'input_ids': encoded_input['input_ids'][i:i+bs].cuda(),
            'attention_mask': encoded_input['attention_mask'][i:i+bs].cuda()
        }
        if i == 0 and verbose:
            print(tokenizer.decode(encoded_input['input_ids'][0].tolist()))
        if plm == 'sbert':
            sentence_embeddings[i:i+bs] = get_sentbert_embed(model, batch_input)
        else:
            sentence_embeddings[i:i+bs] = get_roberta_embed(model, batch_input)

    return sentence_embeddings

def index_nn(out_dir, verbose):
    queries = torch.load(os.path.join(out_dir, 'test_embeds.pt'))   #[1000, 768]
    train = torch.load(os.path.join(out_dir, 'train_embeds.pt')).T  #[768, 1000]

    similarity = queries.mm(train) # [1000_test, 1000_train]
    sorted_nn_ids = torch.argsort(similarity, dim=1, descending=True)
    torch.save(sorted_nn_ids, os.path.join(out_dir, 'sorted_nn_ids.pt'))

    # How good is the sent embed?
    if verbose:
        k = 4
        if 'glue-sst2' in out_dir or 'boolq' in out_dir or 'subj' in out_dir:
            print((sorted_nn_ids[:500,:k] < 500).numpy().mean())
        elif 'scicite' in out_dir:
            print((sorted_nn_ids[:333,:k] < 333).numpy().mean())
        elif 'ag_news' in out_dir:
            print((sorted_nn_ids[:250,:k] < 250).numpy().mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None, required=True)
    parser.add_argument("--plm", type=str, default='sbert', choices=['sbert', 'roberta-large'])
    parser.add_argument("--verbose", default=False, action="store_true")
    args = parser.parse_args()

    out_dir = os.path.join(OUT_SENT_EMBED, args.plm, args.task)
    os.makedirs(out_dir, exist_ok = True)

    # config
    config_map = {'sbert': ("sentence-transformers/all-mpnet-base-v2", 768), 
                    'roberta-large': ("roberta-large", 1024)}
    ckpt, embed_dim = config_map[args.plm]

    # load plm
    tokenizer = AutoTokenizer.from_pretrained(ckpt, cache_dir="cached")
    model = AutoModel.from_pretrained(ckpt, cache_dir="cached")
    model.cuda()

    # encode
    bs = 500
    train_embeds = get_split_embeds(model, tokenizer, args.task, 'train', args.plm, embed_dim, bs, args.verbose)
    test_embeds = get_split_embeds(model, tokenizer, args.task, 'test', args.plm, embed_dim, bs, args.verbose)
    print(train_embeds.shape, test_embeds.shape)

    # dump embeds
    torch.save(train_embeds, os.path.join(out_dir, 'train_embeds.pt'))
    torch.save(test_embeds, os.path.join(out_dir, 'test_embeds.pt'))

    # index by similarity
    index_nn(out_dir, args.verbose)


if __name__ == "__main__":
    main()
