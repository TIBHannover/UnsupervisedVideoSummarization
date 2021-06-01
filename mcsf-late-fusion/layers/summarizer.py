# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable

from .lstmcell import StackedLSTMCell

device = 'cuda' if torch.cuda.is_available() else 'cpu'



class sLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2, m=4, video_type='summe'):
        super().__init__()
        self.csnet_objects = CSNET(input_size, hidden_size, num_layers, m, video_type)
        self.csnet_places = CSNET(input_size, hidden_size, num_layers, m, video_type)

        self.out = nn.Sigmoid()
        if video_type == 'summe':
            self.nframes = 9721
        else:
            self.nframes = 19406
        self.linear = nn.Linear(self.nframes, self.nframes)
        self.linear.weight.data.normal_(0, 1)
        self.fc = nn.Sequential(
            self.linear,
            nn.Sigmoid())

    def forward(self, features, places365_features, difference_attention):
        scores1 = self.csnet_objects(features, difference_attention['objects'])
        scores2 = self.csnet_places(places365_features, difference_attention['places'])
        #late fusion
        res = scores1 + scores2
        return self.out(res)


class CSNET(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2, m=4, video_type='summe'):
        """Scoring LSTM"""
        super().__init__()
        self.m = m
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)  # bidirection => scalar
        self.out = nn.Sigmoid()

        if video_type == 'summe':
            self.nframes = 9721
        else:
            self.nframes = 19406

        self.fc2 = nn.Linear(self.nframes, self.nframes)
        self.fc2.weight.data.normal_(0, 1)
        if self.fc2.bias.data is not None:
            self.fc2.bias.data.zero_()

        self.fc_last = nn.Sequential(
            self.fc2,
            nn.Sigmoid())

    def forward(self, features, difference_attention):
        self.lstm.flatten_parameters()

        # [seq_len, 1, hidden_size * 2]
        # strides stream
        sm_idxs = self.compute_sm(features)
        sm_idxs = self.flatten(list(sm_idxs.values()))

        sm_scores = torch.zeros(features.size(0)).to(device)
        sm = features[sm_idxs]
        sm, (h_n, c_n) = self.lstm(sm)
        sm = self.fc(sm)
        for idx, out in zip(sm_idxs, sm):
            sm_scores[idx] = out

        # chunks stream
        cm_idxs = self.compute_cm(features)
        cm_idxs = self.flatten(list(cm_idxs.values()))

        cm_scores = torch.zeros(features.size(0)).to(device)
        cm = features[cm_idxs]
        cm, (h_n, c_n) = self.lstm(cm)
        cm = self.fc(cm)
        for idx, out in zip(cm_idxs, cm):
            cm_scores[idx] = out
        # cm_scores.unsqueeze_(1)

        difference_attention = difference_attention.squeeze(1)
        # sum scores
        sm_scores = self.out(sm_scores + difference_attention)
        cm_scores = self.out(cm_scores + difference_attention)
        scores = sm_scores + cm_scores
        rest = torch.zeros(int(self.nframes - scores.size(0))).to(device)
        scores = torch.cat((scores, rest))
        self.fc2.weight.data[features.size(0):, :] = torch.zeros(self.fc2.weight.data[features.size(0):, :].size())
        scores = self.fc_last(scores)
        scores = scores[0:features.size(0)]
        return scores.unsqueeze(1)


    # stride streams
    # [Eq. 4]
    def compute_sm(self, image_features):
        T = image_features.size(0)
        M = k = self.m
        sm_idxs = {}
        for m in range(M):
            end = m + T - k
            idxs = []
            for i in range(0, T):
                val = i * k + m
                if val >= end:
                    idxs.append(end)
                    break
                else:
                    idxs.append(val)
            sm_idxs[m] = idxs
        return sm_idxs

    # chunk streams
    # [Eq. 3]
    def compute_cm(self, image_features):
        T = image_features.size(0)
        n_chunks = self.m
        cm_idxs = {}
        for m in range(1, n_chunks + 1):
            fraction = torch.tensor(T / n_chunks)
            start = (m - 1) * torch.ceil(fraction)
            end = m * torch.ceil(fraction) - 1
            idxs = []
            for i in range(T):
                if i >= start and i <= end:
                    idxs.append(i)
            # print('m {}, start {}, end {}'.format(m, start, end))
            # print(idxs)
            cm_idxs[m] = idxs
        return cm_idxs

    def flatten(self, t):
        return [item for sublist in t for item in sublist]


class eLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        """Encoder LSTM"""
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

        self.linear_mu = nn.Linear(hidden_size, hidden_size)
        self.linear_var = nn.Linear(hidden_size, hidden_size)

    def forward(self, frame_features):
        """
        Args:
            frame_features: [seq_len, 1, hidden_size]
        Return:
            last hidden
                h_last [num_layers=2, 1, hidden_size]
                c_last [num_layers=2, 1, hidden_size]
        """
        self.lstm.flatten_parameters()
        _, (h_last, c_last) = self.lstm(frame_features)

        return (h_last, c_last)


class dLSTM(nn.Module):
    def __init__(self, input_size=2048, hidden_size=2048, num_layers=2):
        """Decoder LSTM"""
        super().__init__()

        self.lstm_cell = StackedLSTMCell(num_layers, input_size, hidden_size)
        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, seq_len, init_hidden):
        """
        Args:
            seq_len (int)
            init_hidden
                h [num_layers=2, 1, hidden_size]
                c [num_layers=2, 1, hidden_size]
        Return:
            out_features: [seq_len, 1, hidden_size]
        """

        batch_size = init_hidden[0].size(1)
        hidden_size = init_hidden[0].size(2)

        x = Variable(torch.zeros(batch_size, hidden_size)).to(device)
        h, c = init_hidden  # (h_0, c_0): last state of eLSTM

        out_features = []
        for i in range(seq_len):
            # last_h: [1, hidden_size] (h from last layer)
            # last_c: [1, hidden_size] (c from last layer)
            # h: [2=num_layers, 1, hidden_size] (h from all layers)
            # c: [2=num_layers, 1, hidden_size] (c from all layers)
            (last_h, last_c), (h, c) = self.lstm_cell(x, (h, c))
            x = self.out(last_h)
            out_features.append(last_h)
        # list of seq_len '[1, hidden_size]-sized Variables'
        return out_features


class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.e_lstm = eLSTM(input_size, hidden_size, num_layers)
        self.d_lstm = dLSTM(input_size, hidden_size, num_layers)

        self.softplus = nn.Softplus()

    def reparameterize(self, mu, log_variance):
        """Sample z via reparameterization trick
        Args:
            mu: [num_layers, hidden_size]
            log_var: [num_layers, hidden_size]
        Return:
            h: [num_layers, 1, hidden_size]
        """
        std = torch.exp(0.5 * log_variance)

        # e ~ N(0,1)
        epsilon = Variable(torch.randn(std.size())).to(device)

        # [num_layers, 1, hidden_size]
        return (mu + epsilon * std).unsqueeze(1)

    def forward(self, features):
        """
        Args:
            features: [seq_len, 1, hidden_size]
        Return:
            h: [2=num_layers, 1, hidden_size]
            decoded_features: [seq_len, 1, 2048]
        """
        seq_len = features.size(0)

        # [num_layers, 1, hidden_size]
        h, c = self.e_lstm(features)

        # [num_layers, hidden_size]
        h = h.squeeze(1)

        # [num_layers, hidden_size]
        h_mu = self.e_lstm.linear_mu(h)
        h_log_variance = torch.log(self.softplus(self.e_lstm.linear_var(h)))

        # [num_layers, 1, hidden_size]
        h = self.reparameterize(h_mu, h_log_variance)

        # [seq_len, 1, hidden_size]
        decoded_features = self.d_lstm(seq_len, init_hidden=(h, c))

        # [seq_len, 1, hidden_size]
        # reverse
        decoded_features.reverse()
        decoded_features = torch.stack(decoded_features)
        return h_mu, h_log_variance, decoded_features


class Summarizer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, m=4, video_type='summe'):
        super().__init__()
        self.s_lstm = sLSTM(input_size, hidden_size, num_layers, m, video_type)
        # self.csnet = CSNET(input_size, hidden_size, num_layers)
        self.vae = VAE(input_size, hidden_size, num_layers)

    def forward(self, image_features, places365_features, difference_attention, uniform=False):
        # Apply weights
        if not uniform:
            # [seq_len, 1]
            scores = self.s_lstm(image_features, places365_features, difference_attention)
            # scores = self.csnet(image_features, difference_attention)
            # print(scores)

            # [seq_len, 1, hidden_size]
            weighted_features = image_features * scores.view(-1, 1, 1)
        else:
            scores = None
            weighted_features = image_features

        h_mu, h_log_variance, decoded_features = self.vae(weighted_features)

        return scores, h_mu, h_log_variance, decoded_features


if __name__ == '__main__':
    pass
