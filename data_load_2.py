'''
An entry or sent looks like ...

SOCCER NN B-NP O
- : O O
JAPAN NNP B-NP B-LOC
GET VB B-VP O
LUCKY NNP B-NP O
WIN NNP I-NP O
, , O O
CHINA NNP B-NP B-PER
IN IN B-PP O
SURPRISE DT B-NP O
DEFEAT NN I-NP O
. . O O

Each mini-batch returns the followings:
words: list of input sents. ["The 26-year-old ...", ...]
x: encoded input sents. [N, T]. int64.
is_heads: list of head markers. [[1, 1, 0, ...], [...]]
tags: list of tags.['O O B-MISC ...', '...']
y: encoded tags. [N, T]. int64
seqlens: list of seqlens. [45, 49, 10, 50, ...]
'''
import numpy as np
import torch
from torch.utils import data

from pytorch_pretrained_bert import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
VOCAB = ('<PAD>', 'O', 'B-ENTITY', 'I-ENTITY', '<START>', '<STOP>', 'B-DATE', 'I-DATE', 'B-NUMBER', 'I-NUMBER', 'B-METRIC', 'I-METRIC', 'X')
START_TAG = '<START>'
STOP_TAG = '<STOP>'
tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}

class NerDataset(data.Dataset):
    def __init__(self, entries):
        """
        entries: [([word1, word2,.....], [tags_1,tag_2,.....])]
        """
        sents, tags_li = [], [] # list of lists
        for entry in entries:
            words = entry[0]
            tags = entry[1]
            sents.append(["[CLS]"] + words + ["[SEP]"])
            tags_li.append(["<START>"] + tags + ["<STOP>"])  # TODO
            #if '\\u200b' in words:
            #    print(entry)
        self.sents, self.tags_li = sents, tags_li

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx] # words, tags: string list

        # We give credits only to the first piece.
        x, y = [], [] # list of ids
        is_heads = [] # list. 1: the token is the first piece of a word
        for w, t in zip(words, tags):
            tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = tokenizer.convert_tokens_to_ids(tokens)

            is_head = [1] + [0]*(len(tokens) - 1)

            t = [t] + ["X"] * (len(tokens) - 1)  # X label, no decision
            yy = [tag2idx[each] for each in t]  # (T,)

            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)
        
        #if not (len(x)==len(y)==len(is_heads)):
        #    print(words)
        #    print(tokens)
        assert len(x)==len(y)==len(is_heads), f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}"

        # seqlen
        seqlen = len(y)

        # to string
        words = " ".join(words)
        tags = " ".join(tags)
        return words, x, is_heads, tags, y, seqlen


def pad(batch):
    '''Pads to the longest sample'''
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]  # 0: <pad>
    x = f(1, maxlen)
    y = f(-2, maxlen)


    f = torch.LongTensor

    return words, f(x), is_heads, tags, f(y), seqlens


