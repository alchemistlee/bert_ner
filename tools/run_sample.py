import sys
import os
import random
import argparse

def _read_txt(reqfile, mask_tags = None):
    f = open(reqfile, 'r')
    entries = f.read().strip().split("\n\n")
    res = []
    for entry in entries:
        sens = []
        for line in entry.splitlines():
            if "\\u200b" in line or '\\ufeff' in line:
                continue
            array = line.strip().split(" ")
            if len(array) != 2:
                continue
            if mask_tags and array[1] in mask_tags:
                sens.append('{0} {1}'.format(array[0], 'O'))
            else:
                sens.append(line.strip())
        if sens and len(sens) > 0:
            res.append("\n".join(sens))
    f.close()
    return res

def write_file(reqfile, res):
    with open(reqfile, 'w') as f:
        f.write("\n\n".join(res))

def _dump_data(entries, prefix = ''):
    '''
    train: 90%, valid: 5%, test: 5%
    '''
    def sample():
        rand = random.random()
        if rand < 0.05:
            return 'valid'
        elif rand < 0.1:
            return 'test'
        else:
            return 'train'
   
    train_set = []
    valid_set = []
    test_set = []
    for entry in entries:
        name = sample()
        if name == 'train':
            train_set.append(entry)
        if name == 'valid':
            valid_set.append(entry)
        if name == 'test':
            test_set.append(entry)
    
    random.shuffle(train_set) 
    random.shuffle(valid_set) 
    random.shuffle(test_set) 
    
    write_file('data/{0}train.txt'.format(prefix), train_set)
    write_file('data/{0}test.txt'.format(prefix), test_set)
    write_file('data/{0}valid.txt'.format(prefix), valid_set)

def run_samples(reqfile, mask_tags = None, prefix = ''):
    entries = _read_txt(reqfile, mask_tags)
    _dump_data(entries, prefix)

def shuffle_samples(req_file_lst, out_file):
    res = []
    for req_file in req_file_lst:
        res.extend(_read_txt(req_file))
    random.shuffle(res)
    write_file(out_file, res)

if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--ner_data_path", type=str, default="./")
    hp = parser.parse_args()
    run_samples(hp.ner_data_path, prefix='new_')
    '''
    #req_file_lst = ['data/new_test.txt','data/test.txt']
    #out_file = 'data/merge_test.txt'
    #req_file_lst = ['data/new_valid.txt','data/valid.txt']
    #out_file = 'data/merge_valid.txt'
    req_file_lst = ['data/train.txt','data/new_train.txt']
    out_file = 'data/merge_train.txt'
    shuffle_samples(req_file_lst, out_file)
