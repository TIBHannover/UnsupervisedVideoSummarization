# -*- coding: utf-8 -*-
import h5py
import numpy as np
import torch
import torch.nn as nn
import pickle
import torch.optim as optim
from layers.csnet import CSNET
from tqdm import tqdm, trange

PROCESSED_SUMME = 'data/SumMe/eccv16_dataset_summe_google_pool5.h5'
PROCESSED_TVSUM = 'data/TVSUM/eccv16_dataset_tvsum_google_pool5.h5'

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def save_pickle_file(filename, data):
    print('Saving {} ...'.format(filename))
    with open('{}.pickle'.format(filename), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('{} saved'.format(filename))


class CSNET(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2):
        """Scoring LSTM"""
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, bidirectional=True)
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),  # bidirection => scalar
            nn.Sigmoid())

    def forward(self, feature):
        """
        Args:
            features: [seq_len, 1, 100] (compressed pool5 features)
        Return:
            scores [seq_len, 1]
        """
        self.lstm.flatten_parameters()

        # [seq_len, 1, hidden_size * 2]
        features, (h_n, c_n) = self.lstm(features)

        # [seq_len, 1]
        scores = self.out(features.squeeze(1))

        return scores


class Runner(object):
    def __init__(self, config=None, train_loader=None, test_loader=None):
        """Class that Builds, Trains and Evaluates SUM-GAN model"""
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

    def build(self):

        # Build Modules
        self.csnet = CSNET(
            self.config.input_size,
            self.config.hidden_size
        ).to(device)
        self.model = nn.ModuleList([
            self.csnet])

        self.optimizer = optim.Adam(
            self.csnet.parameters(),
            lr=self.config.lr)

        self.model.train()

        print(self.model)

    def difference(self, h_origin, h_fake):
        return torch.abs(h_origin - h_fake)

    def train(self):
        step = 0
        for epoch_i in trange(self.config.n_epochs, desc='Epoch', ncols=80):
            s_e_loss_history = []
            d_loss_history = []
            c_loss_history = []
            for batch_i, image_features in enumerate(tqdm(
                    self.train_loader, desc='Batch', ncols=80, leave=False)):

                if image_features.size(1) > 10000:
                    continue
                image_features = image_features.view(-1,
                                                     self.config.input_size)

                image_features_ = image_features.to(device)

                T = image_features_.size(0)
                m = 2
                M = k = 4
                cm_idx = {}
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
                    cm_idx[m] = idxs

            # ---- Train sLSTM, eLSTM ----#
            if self.config.verbose:
                tqdm.write('\nTraining sLSTM and eLSTM...')

            # [seq_len, 1, hidden_size]
            # original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)

            self.csnet(
                original_features)
            _, _, _, uniform_features = self.summarizer(
                original_features, uniform=True)

            self.s_e_optimizer.zero_grad()
            s_e_loss.backward()  # retain_graph=True)
            # Gradient cliping
            torch.nn.utils.clip_grad_norm(
                self.model.parameters(), self.config.clip)
            self.s_e_optimizer.step()

            s_e_loss_history.append(s_e_loss.data)

            if self.config.verbose:
                tqdm.write('Plotting...')

            # self.writer.update_loss(uniform_prob.data, step, 'uniform_prob')

            step += 1

        s_e_loss = torch.stack(s_e_loss_history).mean()

        # Plot
        if self.config.verbose:
            tqdm.write('Plotting...')
        self.writer.update_loss(s_e_loss, epoch_i, 's_e_loss_epoch')

        # Save parameters at checkpoint
        ckpt_path = str(self.config.save_dir) + f'/epoch-{epoch_i}.pkl'
        tqdm.write(f'Save parameters at {ckpt_path}')
        if not os.path.exists(self.config.save_dir):
            os.makedirs(self.config.save_dir)
        torch.save(self.model.state_dict(), ckpt_path)
        self.evaluate(epoch_i)

        self.model.train()


if __name__ == '__main__':
    pass
