import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel
from utils import argmax, log_sum_exp
from data_load import START_TAG, STOP_TAG, tag2idx
from torchcrf import CRF

class Net(nn.Module):
    def __init__(self, top_rnns=False, vocab_size=None, device='cpu', finetuning=False, dropout=0., lin_dim=128):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')

        self.top_rnns=top_rnns
        if top_rnns:
            self.rnn = nn.LSTM(bidirectional=True, num_layers=2,
                               input_size=768, hidden_size=lin_dim // 2,
                               batch_first=True, dropout=dropout)
        self.fc = nn.Linear(lin_dim, vocab_size)

        self.vocab_size = vocab_size

        self.device = device
        self.finetuning = finetuning

    def forward(self, x, mask=None):
        '''
        x: (N, T). int64
        y: (N, T). int64

        Returns
        enc: (N, T, VOCAB)
        '''

        if mask is not None:
            mask = mask.to(self.device)

        x = x.to(self.device)

        if self.training and self.finetuning:
            # print("->bert.train()")
            self.bert.train()
            encoded_layers, _ = self.bert(x)
            enc = encoded_layers[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, _ = self.bert(x)
                enc = encoded_layers[-1]

        if self.top_rnns:
            enc, _ = self.rnn(enc)

        if mask is not None:
            enc = enc.masked_fill(mask, 0)

        logits = self.fc(enc)
        y_hat = logits.argmax(-1)
        return logits, y_hat

    def compute_loss(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)

        logits, _ = self.forward(x, mask=y.eq(0).unsqueeze(-1))
        logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
        y = y.view(-1)

        loss = F.cross_entropy(logits, y, ignore_index=0)
        return loss


class NetCRF(Net):
    def __init__(self, *args, **kwargs):
        super(NetCRF, self).__init__(*args, **kwargs)
        self.crf = CRF(self.vocab_size, batch_first=True)

    def compute_loss(self, x, y, reduction='token_mean'):
        y = y.to(self.device)
        feats, _ = super(NetCRF, self).forward(x, mask=y.eq(0).unsqueeze(-1))
        return - self.crf(feats, y, mask=y.ne(0), reduction=reduction)
        #return - self.crf(feats, y, mask=y.ne(0), reduction='none')

    def forward(self, x, y):  # TODO replace y with mask
        return self._decode(x, y)

    def _decode(self, x, y):
        y = y.to(self.device)
        feats, _ = super(NetCRF, self).forward(x, mask=y.eq(0).unsqueeze(-1))
        y_hat = self.crf.decode(feats, mask=y.ne(0))
        return feats, y_hat


class NetCRFDouble(Net):
    def __init__(self, *args, **kwargs):
        super(NetCRFDouble, self).__init__(*args, **kwargs)
        self.crf1 = CRF(self.vocab_size, batch_first=True)
        self.crf2 = CRF(self.vocab_size, batch_first=True)

    def compute_loss(self, x, y, id, reduction='token_mean'):
        y = y.to(self.device)
        feats, _ = super(NetCRFDouble, self).forward(x, mask=y.eq(0).unsqueeze(-1))
        if id in [1]:
            return - self.crf1(feats, y, mask=y.ne(0), reduction=reduction)
        elif id == 2:
            return - self.crf2(feats, y, mask=y.ne(0), reduction=reduction)

    def forward(self, x, y, id):  # TODO replace y with mask
        return self._decode(x, y, id)

    def _decode(self, x, y, id):
        y = y.to(self.device)
        feats, _ = super(NetCRFDouble, self).forward(x, mask=y.eq(0).unsqueeze(-1))
        if id in [1]:
            y_hat = self.crf1.decode(feats, mask=y.ne(0))
        elif id == 2:
            y_hat = self.crf2.decode(feats, mask=y.ne(0))

        return feats, y_hat
