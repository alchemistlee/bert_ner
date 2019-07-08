import os
import sys
from collections import ChainMap

def _read_txt(reqfile):
    f = open(reqfile, 'r')
    sen_mp = {}
    entries = f.read().strip().split("\n\n")
    for entry in entries:
        tokens = []
        for line in entry.splitlines():
            array = line.strip().split(" ")
            if len(array) != 2:
                continue
            tokens.append(array[0]) 
        sen = ''.join(tokens)
        sen_mp[sen] = entry
    f.close()
    return sen_mp

def _merge(req_file_1, req_file_2):
    return ChainMap(_read_txt(req_file_1), _read_txt(req_file_2))

def _overlap(ori_map, cur_map):
    overlap_keys = []
    for key in ori_map:
        if key in cur_map:
            overlap_keys.append(key)
    return overlap_keys

if __name__ == "__main__":
    ori_mp = _merge('ori_valid.txt', 'ori_test.txt')
    cur_mp = _merge('cur_valid.txt', 'cur_test.txt')
    overlap_keys = _overlap(ori_mp, cur_mp)
    print('overlap keys size = {0}'.format(len(overlap_keys)))
    ori_out_file, cur_out_file = 'ori_overlap.txt', 'cur_overlap.txt'
    ori_out, cur_out = [], []
    # write files
    for key in overlap_keys:
        ori_out.append(ori_mp[key])
        cur_out.append(cur_mp[key])

    with open(ori_out_file, 'w') as f:
        f.write("\n\n".join(ori_out))
    with open(cur_out_file, 'w') as f:
        f.write("\n\n".join(cur_out))
