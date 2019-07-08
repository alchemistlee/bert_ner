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


def train(model, iterator_list, optimizer):
    model.train()
    
    data_loader_1 = iterator_list[0]
    iter_1 = iter(data_loader_1)

    vars = locals()

    def grab_data(id):
        assert(id in [1])
        try:
            words, x, is_heads, tags, y, seqlens = trunk_batch(
                    next(vars[f'iter_{id}']))
        except StopIteration:
            vars[f'iter_{id}'] = iter(vars[f'data_loader_{id}'])
            words, x, is_heads, tags, y, seqlens = trunk_batch(
                    next(vars[f'iter_{id}']))
        return words, x, is_heads, tags, y, seqlens

    for i in range(len(data_loader_1)):
        optimizer.zero_grad()

        words, x, is_heads, tags, y, seqlens = grab_data(1)
        loss1 = model.compute_loss(x, y)

        loss = loss1 
        loss.backward()
        optimizer.step()

        if i == 0:
            print("=====sanity check======")
            print("words:", words[0])
            print("x:", x.cpu().numpy()[0][:seqlens[0]])
            print("tokens:", tokenizer.convert_ids_to_tokens(x.cpu().numpy()[0])[:seqlens[0]])
            print("is_heads:", is_heads[0])
            print("tags:", tags[0])
            print("seqlen:", seqlens[0])
            print("=======================")

        if i % 10 == 0: # monitoring
            print(f"step: {i}, loss1: {loss1.item()}, loss: {loss.item()}")


def eval(model, iterator, f):
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = trunk_batch(batch)

            if getattr(model, 'crf', None) is None:
                _, y_hat = model(x)  # y_hat: (N, T)
            else:
                _, y_hat = model(x, y)  # y_hat: (N, T)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist() if isinstance(y_hat, torch.Tensor) else y_hat)

    tmp_fname = "{0}.temp".format(f)
    ## gets results and save
    with open(tmp_fname, 'w') as fout:
        for rid, (words, is_heads, tags, y_hat) in enumerate(zip(Words, Is_heads, Tags, Y_hat)):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
            try:
                assert len(preds) == len(words.split()) == len(tags.split())
            except AssertionError:
                print(f'Skipping {rid}')
                continue
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                fout.write(f"{w} {t} {p}\n")
            fout.write("\n")
    ## calc metric
    y_true =  np.array([tag2idx[line.split()[1]] for line in open(tmp_fname, 'r').read().splitlines() if len(line) > 0])
    y_pred =  np.array([tag2idx[line.split()[2]] for line in open(tmp_fname, 'r').read().splitlines() if len(line) > 0])

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

    final = f + ".P%.4f_R%.4f_F%.4f" %(precision, recall, f1)
    with open(final, 'w') as fout:
        result = open(tmp_fname, "r").read()
        fout.write(f"{result}\n")

        fout.write(f"precision={precision}\n")
        fout.write(f"recall={recall}\n")
        fout.write(f"f1={f1}\n")

    os.remove(tmp_fname)

    print("precision=%.4f"%precision)
    print("recall=%.4f"%recall)
    print("f1=%.4f"%f1)
    return precision, recall, f1

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--top_rnns", dest="top_rnns", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/01")
    parser.add_argument("--trainset_1", type=str, default="jiashi/merge_train.txt")
    parser.add_argument("--validset", type=str, default="jiashi/merge_valid.txt")
    parser.add_argument("--model", type=str, default="crf")
    parser.add_argument("--lr_decay_pat", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--lin_dim", type=int, default=128)
    hp = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    if hp.model == "crf":
        model = NetCRF(hp.top_rnns, len(VOCAB), device, hp.finetuning, dropout=hp.dropout, lin_dim = hp.lin_dim).cuda()
    elif hp.model == "crf_double":
        model = NetCRFDouble(hp.top_rnns, len(VOCAB), device, hp.finetuning, dropout=hp.dropout).cuda()
    else:
        model = Net(hp.top_rnns, len(VOCAB), device, hp.finetuning, dropout=hp.dropout).cuda()
    #model = nn.DataParallel(model)

    train_dataset_1 = NerDataset(hp.trainset_1)
    data_loader_1 = data.DataLoader(dataset=train_dataset_1,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)

    eval_dataset = NerDataset(hp.validset)
    eval_iter = data.DataLoader(dataset=eval_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)

    optimizer = optim.Adam(model.parameters(), lr = hp.lr)

    best_f1 = 0
    lr_decay_count = best_f1 = 0
    for epoch in range(1, hp.n_epochs+1):
        train(model, [data_loader_1], optimizer)

        print(f"=========eval at epoch={epoch}=========")
        if not os.path.exists(hp.logdir): os.makedirs(hp.logdir)
        fname = os.path.join(hp.logdir, 'best')
        precision, recall, f1 = eval(model, eval_iter, fname)

        # train eval..
        print("TRAINING SET Eval: ")
        tr_precision, tr_recall, tr_f1 = eval(model, iter(data_loader_1), fname+'-tr1')

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f"{fname}.pt")
            print(f"INFO: weights were saved to {fname}.pt")
            lr_decay_count = 0
        else:
            lr_decay_count += 1
            if lr_decay_count == hp.lr_decay_pat:
                print('INFO: learning rate decay...')
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 2
                    cur_lr = param_group['lr']
                lr_decay_count = 0
                # early stopping
                if cur_lr < 1e-6:
                    print(f"INFO: early stopping at {epoch}")
                    break
