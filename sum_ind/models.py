import torch
import torch.nn as nn
from torch.nn import functional as F
from indrnn import IndRNNv2

__all__ = ['DSN','SumInd']

class DSN(nn.Module):
    """Deep Summarization Network"""
    def __init__(self, in_dim=1024, hid_dim=256, num_layers=1, cell='lstm'):
        super(DSN, self).__init__()
        assert cell in ['lstm', 'gru'], "cell must be either 'lstm' or 'gru'"
        if cell == 'lstm':
            self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.GRU(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hid_dim*2, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        p = F.sigmoid(self.fc(h))
        return p

class SumInd(nn.Module):
    """Deep Summarization Network"""

    def __init__(self, in_dim=1024, hid_dim=128, num_layers=2):
        super(SumInd, self).__init__()
        # Parameters taken from https://arxiv.org/abs/1803.04831
        self.TIME_STEPS = in_dim
        self.RECURRENT_MAX = pow(2, 1 / self.TIME_STEPS)
        self.RECURRENT_MIN = pow(1 / 2, 1 / self.TIME_STEPS)
        self.bidirectional=True

        recurrent_inits = []
        for _ in range(num_layers - 1):
            recurrent_inits.append(
                lambda w: nn.init.uniform_(w, 0, self.RECURRENT_MAX)
            )
        recurrent_inits.append(lambda w: nn.init.uniform_(
            w, self.RECURRENT_MIN, self.RECURRENT_MAX))
        self.indrnn = IndRNNv2(
            in_dim, hid_dim, num_layers, batch_norm=True,
            hidden_max_abs=self.RECURRENT_MAX, batch_first=True,
            bidirectional=self.bidirectional, recurrent_inits=recurrent_inits, nonlinearity="relu",
            gradient_clip=5
        )

        self.lin = nn.Linear(
            hid_dim * 2 if self.bidirectional else hid_dim, 1)
        self.lin.bias.data.fill_(.1)
        self.lin.weight.data.normal_(0, .01)

    def forward(self, x):
        h, _ = self.indrnn(x)  # h.shape : [seq_len,batch_size,hid_dim].. here batch sie is #frames in a sequence
        p = torch.sigmoid(self.lin(h))  # p.shape : [seq_len,batch_size,1] 1 output for each input aka frame
        return p
