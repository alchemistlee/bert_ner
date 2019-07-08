import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from model import Net, NetCRF, NetCRFDouble
from data_load_2 import NerDataset, pad, VOCAB, tokenizer, tag2idx, idx2tag
import os
import numpy as np
import argparse
from utils import trunk_batch, pre_process
from random import choice
import re
from result import Result

def eval(model, iterator):
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat_1, Y_hat_2, X = [], [], [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = trunk_batch(batch)
            
            _, y_hat_1 = model(x, y, 1)  # y_hat: (N, T)
            _, y_hat_2 = model(x, y, 2)  # y_hat: (N, T)
            
            Words.extend(words)
            X.extend(x)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat_1.extend(y_hat_1.cpu().numpy().tolist() if isinstance(y_hat_1, torch.Tensor) else y_hat_1)
            Y_hat_2.extend(y_hat_2.cpu().numpy().tolist() if isinstance(y_hat_2, torch.Tensor) else y_hat_2)

    res = []
    for rid, (words, is_heads, tags, y_hat_1, y_hat_2, x) in enumerate(zip(Words, Is_heads, Tags, Y_hat_1, Y_hat_2, X)):
        y_hat_1 = [hat for head, hat in zip(is_heads, y_hat_1) if head == 1]
        y_hat_2 = [hat for head, hat in zip(is_heads, y_hat_2) if head == 1]
        preds_1 = [idx2tag[hat] for hat in y_hat_1]
        preds_2 = [idx2tag[hat] for hat in y_hat_2]
    
        try:
            assert len(preds_1) == len(words.split()) == len(tags.split()) == len(preds_2)
        except AssertionError:
            print(f'Skipping {rid}')
            continue
        
        best_preds = preds_1
        likelihood_1 = get_likelihood(model, x, y_hat_1, 1)
        likelihood_2 = get_likelihood(model, x, y_hat_2, 2)
        if likelihood_1 < likelihood_2:
            best_preds = preds_2

        eval = Result()  
        res.append(eval.result_to_json(words, tokens, best_preds))
    
    return res

def get_likelihood(model, x, y_hat, id):
    with torch.no_grad():
        neg_logprob = model.compute_loss(x[:len(y_hat)].unsqueeze(0), \
                torch.tensor(y_hat).unsqueeze(0), id, 'none')
    return math.exp(-neg_logprob)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--top_rnns", dest="top_rnns", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/01")
    parser.add_argument("--model", type=str, default="crf")
    parser.add_argument("--lr_decay_pat", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--model_path", type=str, default="./checkpoints/ensemble/best.pt")
    
    hp = parser.parse_args()
    
    torch.manual_seed(1)
    hp.top_rnns = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = NetCRFDouble(hp.top_rnns, len(VOCAB), device, hp.finetuning, dropout=hp.dropout).cuda()
    model.load_state_dict(torch.load(hp.model_path))

    while True:
        content = input("input query: \n")
        if content == "q":
            break
        entries = []
        words = pre_process(content) 
        tags = ["O"] * len(words)
        entries.append((words, tags))
        
        eval_dataset = NerDataset(entries)
        eval_iter = data.DataLoader(dataset=eval_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=1,
                                collate_fn=pad)

        res = eval(model, eval_iter)
        print(res)
