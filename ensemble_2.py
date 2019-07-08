import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from model import Net, NetCRF, NetCRFDouble
from data_load import NerDataset, pad, VOCAB, tokenizer, tag2idx, idx2tag
import os
import numpy as np
import argparse
from utils import trunk_batch
from random import choice
import math
from itertools import product
import datetime

def eval(model, iterator, f, alpha):
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
    
    req_file = '{0}.temp.{1}'.format(f, alpha)
    fout = open(req_file, 'w')
    # sample by sample
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

        for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], best_preds[1:-1]):
            fout.write(f"{w} {t} {p}\n")
        fout.write("\n")
    fout.close()
    
    # cal metric
    return _cal_metric(req_file, tag2idx, f, alpha) 

def _cal_metric(req_file, tag2idx, f, alpha):
    y_true =  np.array([tag2idx[line.split()[1]] for line in open(req_file, 'r').read().splitlines() if len(line) > 0])
    y_pred =  np.array([tag2idx[line.split()[2]] for line in open(req_file, 'r').read().splitlines() if len(line) > 0])

    num_proposed = len(y_pred[y_pred>1])
    num_correct = (np.logical_and(y_true==y_pred, y_true>1)).astype(np.int).sum()
    num_gold = len(y_true[y_true>1])

    print(f"num_proposed:{num_proposed}")
    print(f"num_correct:{num_correct}")
    print(f"num_gold:{num_gold}")
    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        if precision*recall==0:
            f1=1.0
        else:
            f1=0

    final = f + ".P%.4f_R%.4f_F%.4f_alpha_%.2f" %(precision, recall, f1, alpha)
    with open(final, 'w') as fout:
        result = open(req_file, "r").read()
        fout.write(f"{result}\n")

        fout.write(f"precision={precision}\n")
        fout.write(f"recall={recall}\n")
        fout.write(f"f1={f1}\n")

    os.remove(req_file)

    print("precision=%.4f"%precision)
    print("recall=%.4f"%recall)
    print("f1=%.4f"%f1)
    return precision, recall, f1


def get_likelihood(model, x, y_hat, id):
    with torch.no_grad():
        neg_logprob = model.compute_loss(x[:len(y_hat)].unsqueeze(0), \
                torch.tensor(y_hat).unsqueeze(0), id, 'none')
    return math.exp(-neg_logprob)

def violent_combine(y_hat_1, y_hat_2):
    tag_ids = []
    for i in range(len(y_hat_1)):
        sub_tag_ids = set()
        sub_tag_ids.add(y_hat_1[i])
        sub_tag_ids.add(y_hat_2[i])
        tag_ids.append(sub_tag_ids)
    # combine
    candidates = []
    for ids in product(*tag_ids):
        candidates.append(list(ids))
    return candidates

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--top_rnns", dest="top_rnns", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/test")
    parser.add_argument("--trainset_1", type=str, default="jiashi/brat.txt")
    parser.add_argument("--trainset_2", type=str, default="jiashi/test_20190620.txt")
    parser.add_argument("--validset", type=str, default="jiashi/merge_test.txt")
    parser.add_argument("--model", type=str, default="crf_double")
    parser.add_argument("--lr_decay_pat", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--lin_dim", type=int, default=128)
    #parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--model_path", type=str, default="checkpoints/ensemble/best.pt")
    hp = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    hp.top_rnns = True #default
    model = NetCRFDouble(hp.top_rnns, len(VOCAB), device, hp.finetuning, dropout=hp.dropout).cuda()
    model.load_state_dict(torch.load(hp.model_path))
    
    eval_dataset = NerDataset(hp.validset)
    eval_iter = data.DataLoader(dataset=eval_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)
    
    fname = os.path.join(hp.logdir, 'best')
    for i in range(11):
        alpha = i * 0.1
    #alpha = hp.alpha
        starttime = datetime.datetime.now()
        precision, recall, f1 = eval(model, eval_iter, fname, alpha)
        endtime = datetime.datetime.now()
        print('alpha [{0}], precision [{1}], recall [{2}], f1 [{3}, cost [{4}]'.format(alpha, precision, recall, f1, (endtime-starttime).seconds))
