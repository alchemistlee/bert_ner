import torch
import re

MAX_SEQ_LENGTH = 512
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def trunk_batch(batch):
    words, x, is_heads, tags, y, seqlens = batch

    if x.size(1) > MAX_SEQ_LENGTH:
        x = x[:, : MAX_SEQ_LENGTH]
        is_heads = list(map(lambda x: x[: MAX_SEQ_LENGTH], is_heads))

        _tags = []
        for t in tags:
            _t = t.split()
            if len(_t) > MAX_SEQ_LENGTH:
                _t = _t[: MAX_SEQ_LENGTH]
            _tags.append(' '.join(_t))
        tags = _tags
        
        _words = []
        for w in words:
            _w = w.split()
            if len(_w) > MAX_SEQ_LENGTH:
                _w = _w[: MAX_SEQ_LENGTH]
            _words.append(' '.join(_w))
        words = _words

        y = y[:, : MAX_SEQ_LENGTH]

        for i, elem in enumerate(seqlens):
            if elem > MAX_SEQ_LENGTH:
                seqlens[i] = MAX_SEQ_LENGTH
    return words, x, is_heads, tags, y, seqlens

def pre_process(query):
    def is_en(uchar):
        if (uchar >= u'\u0041' and uchar<=u'\u005a') \
            or (uchar >= u'\u0061' and uchar<=u'\u007a'):
            return True
        return False
       
    def is_digit(c):
        value = re.compile(r'^[-+]?[0-9%\\.]+$')
        result = value.match(c)
        if result:
            return True
        return False

    res = []
    prev_value = ''
    for c in list(query):
        if is_en(c):
            if prev_value == '':
                prev_value = c
                continue
            if is_en(prev_value):
                prev_value += c
            else:
                res.append(prev_value)
                prev_value = c
        elif is_digit(c):
            if prev_value == '':
                prev_value = c
                continue
            if is_digit(prev_value):
                prev_value += c
            else:
                res.append(prev_value)
                prev_value = c
        else:
            if prev_value != '':
                res.append(prev_value)
            if c == ' ':
                prev_value = ''
                continue
            prev_value = c
    if prev_value != '':
        res.append(prev_value)
    return res

