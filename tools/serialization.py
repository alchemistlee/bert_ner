import os
import sys
from result import Result

def read_txt(req_file):
    f = open(req_file, 'r') 
    entries = f.read().strip().split("\n\n")
    tokens_lst, gt_lst, pred_lst = [], [], []
    for entry in entries:
        tokens, gt, pred = [], [], []
        for line in entry.splitlines():
            array = line.strip().split(" ")
            if len(array) != 3:
                continue
            tokens.append(array[0]) 
            gt.append(array[1])
            pred.append(array[2])
        if gt == pred:
            continue
        tokens_lst.append(tokens)
        gt_lst.append(gt)
        pred_lst.append(pred)
    return tokens_lst, gt_lst, pred_lst

def serialize(tokens, tags):
    eval = Result()
    mp = eval.get_result(tokens, tags)
    return mp

if __name__ == "__main__":
    tokens_lst, gt_lst, pred_lst = read_txt('../checkpoints/ensemble/best.P0.8935_R0.8706_F0.8819') 
    for tokens, gt, pred in zip(tokens_lst, gt_lst, pred_lst):
        print("".join(tokens))
        mp = serialize(tokens, gt)
        print(mp)
        mp = serialize(tokens, pred)
        print(mp)
        print("\n")
