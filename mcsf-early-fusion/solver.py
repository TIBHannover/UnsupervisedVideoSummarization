# -*- coding: utf-8 -*-
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import json
from tqdm import tqdm, trange
import os

from layers import Summarizer, Discriminator  # , apply_weight_norm
from utils import TensorboardWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Solver(object):
    def __init__(self, config=None, train_loader=None, test_loader=None, difference_attention=None):
        """Class that Builds, Trains and Evaluates SUM-GAN model"""
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.difference_attention = difference_attention

    def build(self):

        # Build Modules
        self.linear_compress = nn.Linear(
            self.config.input_size,
            self.config.hidden_size).to(device)
        self.summarizer = Summarizer(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            m=self.config.m,
            video_type=self.config.video_type).to(device)
        self.discriminator = Discriminator(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers).to(device)
        self.model = nn.ModuleList([
            self.linear_compress, self.summarizer, self.discriminator])

        if self.config.mode == 'train':
            # Build Optimizers
            self.s_e_optimizer = optim.Adam(
                list(self.summarizer.s_lstm.parameters())
                + list(self.summarizer.vae.e_lstm.parameters())
                + list(self.linear_compress.parameters()),
                lr=self.config.lr)
            self.d_optimizer = optim.Adam(
                list(self.summarizer.vae.d_lstm.parameters())
                + list(self.linear_compress.parameters()),
                lr=self.config.lr)
            self.c_optimizer = optim.Adam(
                list(self.discriminator.parameters())
                + list(self.linear_compress.parameters()),
                lr=self.config.discriminator_lr)

            self.s_e_scheduler = StepLR(
                self.s_e_optimizer, step_size=10, gamma=0.1)
            self.d_scheduler = StepLR(
                self.d_optimizer, step_size=10, gamma=0.1)
            self.c_scheduler = StepLR(
                self.c_optimizer, step_size=10, gamma=0.1)

            # self.model.train()

            # Tensorboard
            self.writer = TensorboardWriter(self.config.log_dir)

    @staticmethod
    def freeze_model(module):
        for p in module.parameters():
            p.requires_grad = False

    def reconstruction_loss(self, h_origin, h_fake):
        """L2 loss between original-regenerated features at cLSTM's last hidden layer"""

        return torch.norm(h_origin - h_fake, p=2)

    def prior_loss(self, mu, log_variance):
        """KL( q(e|x) || N(0,1) )"""
        return 0.5 * torch.sum(-1 + log_variance.exp() + mu.pow(2) - log_variance)

    def sparsity_loss(self, scores):
        """Summary-Length Regularization"""

        return torch.abs(torch.mean(scores) - self.config.summary_rate)

    def gan_loss(self, original_prob, fake_prob, uniform_prob):
        """Typical GAN loss + Classify uniformly scored features"""

        gan_loss = torch.mean(torch.log(original_prob) + torch.log(1 - fake_prob)
                              + torch.log(1 - uniform_prob))  # Discriminate uniform score

        return gan_loss

    def variance_loss(self, scores, epsilon=1e-4):
        median_tensor = torch.zeros(scores.shape[0]).to(device)
        median_tensor.fill_(torch.median(scores))
        loss = nn.MSELoss()
        variance = loss(scores.squeeze(), median_tensor)
        return 1 / (variance + epsilon)

    def train(self):
        step = 0
        for epoch_i in trange(self.config.n_epochs, desc='Epoch', ncols=80):
            s_e_loss_history = []
            d_loss_history = []
            c_loss_history = []
            print('Epoch: ', epoch_i, ' LR: ',
                  self.s_e_scheduler.get_last_lr())
            for batch_i, data in enumerate(tqdm(
                    self.train_loader, desc='Batch', ncols=80, leave=False)):

                self.model.train()
                image_features = data[0]
                video_name = data[1][0]
                attention_ = torch.tensor(
                    self.difference_attention[video_name]).to(device)
                if image_features.size(1) > 10000:
                    continue

                # [batch_size=1, seq_len, 2048]
                # [seq_len, 1024]
                image_features = image_features.view(-1,
                                                     self.config.input_size)

                # [seq_len, 1024]
                image_features_ = Variable(image_features).to(device)

                # ---- Train sLSTM, eLSTM ----#
                if self.config.verbose:
                    tqdm.write('\nTraining sLSTM and eLSTM...')

                # [seq_len, 1, hidden_size]
                original_features = self.linear_compress(
                    image_features_.detach()).unsqueeze(1)

                scores, h_mu, h_log_variance, generated_features = self.summarizer(
                    original_features, attention_)
                _, _, _, uniform_features = self.summarizer(
                    original_features, attention_, uniform=True)

                h_origin, original_prob = self.discriminator(original_features)
                h_fake, fake_prob = self.discriminator(generated_features)
                h_uniform, uniform_prob = self.discriminator(uniform_features)

                tqdm.write(
                    f'original_p: {original_prob.item():.3f}, fake_p: {fake_prob.item():.3f}, uniform_p: {uniform_prob.item():.3f}')

                reconstruction_loss = self.reconstruction_loss(
                    h_origin, h_fake)
                prior_loss = self.prior_loss(h_mu, h_log_variance)
                sparsity_loss = self.sparsity_loss(scores)
                variance_loss = self.variance_loss(scores)

                tqdm.write(
                    f'recon loss {reconstruction_loss.item():.3f}, prior loss: {prior_loss.item():.3f}, sparsity loss: {sparsity_loss.item():.3f},'
                    f'variance loss: {variance_loss.item():.3f}')

                s_e_loss = reconstruction_loss + prior_loss + sparsity_loss + variance_loss

                self.s_e_optimizer.zero_grad()
                s_e_loss.backward()  # retain_graph=True)
                # Gradient cliping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.clip)
                self.s_e_optimizer.step()

                s_e_loss_history.append(s_e_loss.data)

                # ---- Train dLSTM ----#
                if self.config.verbose:
                    tqdm.write('Training dLSTM...')

                # [seq_len, 1, hidden_size]
                original_features = self.linear_compress(
                    image_features_.detach()).unsqueeze(1)

                scores, h_mu, h_log_variance, generated_features = self.summarizer(
                    original_features, attention_)
                _, _, _, uniform_features = self.summarizer(
                    original_features, attention_, uniform=True)

                h_origin, original_prob = self.discriminator(original_features)
                h_fake, fake_prob = self.discriminator(generated_features)
                h_uniform, uniform_prob = self.discriminator(uniform_features)

                tqdm.write(
                    f'original_p: {original_prob.item():.3f}, fake_p: {fake_prob.item():.3f}, uniform_p: {uniform_prob.item():.3f}')

                reconstruction_loss = self.reconstruction_loss(
                    h_origin, h_fake)
                gan_loss = self.gan_loss(
                    original_prob, fake_prob, uniform_prob)

                tqdm.write(
                    f'recon loss {reconstruction_loss.item():.3f}, gan loss: {gan_loss.item():.3f}')

                d_loss = reconstruction_loss + gan_loss

                self.d_optimizer.zero_grad()
                d_loss.backward()  # retain_graph=True)
                # Gradient cliping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.clip)
                self.d_optimizer.step()

                d_loss_history.append(d_loss.data)

                # ---- Train cLSTM ----#
                if batch_i > self.config.discriminator_slow_start:
                    if self.config.verbose:
                        tqdm.write('Training cLSTM...')
                    # [seq_len, 1, hidden_size]
                    original_features = self.linear_compress(
                        image_features_.detach()).unsqueeze(1)

                    scores, h_mu, h_log_variance, generated_features = self.summarizer(
                        original_features, attention_)
                    _, _, _, uniform_features = self.summarizer(
                        original_features, attention_, uniform=True)

                    h_origin, original_prob = self.discriminator(
                        original_features)
                    h_fake, fake_prob = self.discriminator(generated_features)
                    h_uniform, uniform_prob = self.discriminator(
                        uniform_features)
                    tqdm.write(
                        f'original_p: {original_prob.item():.3f}, fake_p: {fake_prob.item():.3f}, uniform_p: {uniform_prob.item():.3f}')

                    # Maximization
                    c_loss = -1 * \
                        self.gan_loss(original_prob, fake_prob, uniform_prob)

                    tqdm.write(f'gan loss: {gan_loss.item():.3f}')

                    self.c_optimizer.zero_grad()
                    c_loss.backward()
                    # Gradient cliping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.clip)
                    self.c_optimizer.step()
                    c_loss_history.append(c_loss.data)

                if self.config.verbose:
                    tqdm.write('Plotting...')

                self.writer.update_loss(
                    reconstruction_loss.data, step, 'recon_loss')
                self.writer.update_loss(prior_loss.data, step, 'prior_loss')
                self.writer.update_loss(
                    sparsity_loss.data, step, 'sparsity_loss')
                self.writer.update_loss(gan_loss.data, step, 'gan_loss')
                self.writer.update_loss(
                    variance_loss.data, step, 'variance_loss')

                self.writer.update_loss(
                    original_prob.data, step, 'original_prob')
                self.writer.update_loss(fake_prob.data, step, 'fake_prob')
                self.writer.update_loss(
                    uniform_prob.data, step, 'uniform_prob')

                step += 1

            s_e_loss = torch.stack(s_e_loss_history).mean()
            d_loss = torch.stack(d_loss_history).mean()
            c_loss = torch.stack(c_loss_history).mean()

            # Plot
            if self.config.verbose:
                tqdm.write('Plotting...')
            self.writer.update_loss(s_e_loss, epoch_i, 's_e_loss_epoch')
            self.writer.update_loss(d_loss, epoch_i, 'd_loss_epoch')
            self.writer.update_loss(c_loss, epoch_i, 'c_loss_epoch')

            # Save parameters at checkpoint
            ckpt_path = str(self.config.save_dir) + f'/epoch-{epoch_i}.pkl'
            tqdm.write(f'Save parameters at {ckpt_path}')
            if not os.path.exists(self.config.save_dir):
                os.makedirs(self.config.save_dir)
            torch.save(self.model.state_dict(), ckpt_path)
            self.evaluate(epoch_i)
            # self.model.train()

            # update schedulers
            self.s_e_scheduler.step()
            self.d_scheduler.step()
            self.c_scheduler.step()

    def evaluate(self, epoch_i):

        self.model.eval()

        out_dict = {}

        for video_tensor, video_name in tqdm(
                self.test_loader, desc='Evaluate', ncols=80, leave=False):

            # [seq_len, batch=1, 1024]
            video_tensor = video_tensor.view(-1, self.config.input_size)
            video_feature = Variable(video_tensor, volatile=True).to(device)

            # [seq_len, 1, hidden_size]
            video_feature = self.linear_compress(
                video_feature.detach()).unsqueeze(1)
            attention_ = torch.tensor(
                self.difference_attention[video_name]).to(device)

            # [seq_len]
            with torch.no_grad():
                scores = self.summarizer.s_lstm(
                    video_feature, attention_).squeeze(1)
                scores = scores.cpu().detach().numpy().tolist()

                out_dict[video_name] = scores

            score_save_path = self.config.score_dir.joinpath(
                f'{self.config.video_type}_{epoch_i}.json')
            if not os.path.exists(self.config.score_dir):
                os.makedirs(self.config.score_dir)
            with open(score_save_path, 'w') as f:
                tqdm.write(f'Saving score at {str(score_save_path)}.')
                json.dump(out_dict, f)
            score_save_path.chmod(0o777)

    def pretrain(self):
        pass


if __name__ == '__main__':
    pass
