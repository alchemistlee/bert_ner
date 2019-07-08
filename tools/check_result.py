import os
import sys

def combine(entry):
    def get_ent(entry, id):
        prev = -1
        res = []
        ent = []
        for idx, content in enumerate(entry.splitlines()): 
            array = content.strip().split(" ")
            if len(array) != 3:
                break
            if array[id].find('B-ENTITY') != -1:
                ent = []
                prev = idx
                ent.append(array[0])
            elif array[id].find('I-ENTITY') != -1 and idx - prev == 1:
                ent.append(array[0])
                prev = idx
            else:
                if ent and len(ent) > 0:
                    res.append("".join(ent))
                prev = -1
                ent = []
        if ent and len(ent) > 0:
            res.append("".join(ent))
        return res
    return get_ent(entry, 1), get_ent(entry, 2) 

def read_txt(reqfile):
    def print_res(entry, sen, ground, pred):
        #print("{0}\ny = {1}\t pred = {2}\n\nTags\n{3}\n".format("".join(sen), ",".join(ground), ",".join(pred), entry))
        print("{0}\nGT = {1}\t PRED = {2}\n".format("".join(sen), ",".join(ground), ",".join(pred)))

    f = open(reqfile, 'r') 
    entries = f.read().strip().split("\n\n")
    wrong_entries = []
    del[entries[-1]]
    for entry in entries:
        lines = entry.splitlines() 
        for content in lines:
            array = content.strip().split(" ")
            if len(array) != 3 or array[1] != array[2]:
                sen = [content.strip().split()[0] for content in entry.splitlines()]
                ground, pred = combine(entry)
                print_res(entry, sen, ground, pred)
                break

if __name__ == "__main__":
    reqfile = sys.argv[1]
    read_txt(reqfile)
