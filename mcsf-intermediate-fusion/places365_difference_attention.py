# -*- coding: utf-8 -*-
import h5py
import numpy as np
import torch
import torch.nn as nn
import pickle
from os import listdir, path
from utils import drop_file_extension, open_pickle_file

DOWNSAMPLED_SUMME_PLACES365_FEATURES = 'data/plc_365/place365_summe/'
DOWNSAMPLED_TVSUM_PLACES365_FEATURES = 'data/plc_365/place365_tvsum/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def save_pickle_file(filename, data):
    print('Saving {} ...'.format(filename))
    with open('{}.pickle'.format(filename), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('{} saved'.format(filename))


class FC(nn.Module):
    def __init__(self, input_size, output_size=1):
        """Scoring LSTM"""
        super().__init__()
        self.fc_dt1 = nn.Linear(input_size, output_size)
        self.fc_dt2 = nn.Linear(input_size, output_size)
        self.fc_dt4 = nn.Linear(input_size, output_size)

    def forward(self, input, difference=1):
        if difference == 1:
            return self.fc_dt1(input)
        if difference == 2:
            return self.fc_dt2(input)
        if difference == 4:
            return self.fc_dt4(input)


class DifferenceAttention(object):
    def __init__(self, places365_features, input_size=1024, out_file=''):
        """Class that Builds, Trains and Evaluates SUM-GAN model"""
        self.input_size = input_size
        self.places365_features = places365_features
        self.out_file = out_file

    def build(self):
        # Build Modules
        self.fc = FC(
            self.input_size).to(device)
        self.model = nn.ModuleList([self.fc])
        self.model.train()
        print(self.model)

    def difference(self, h_origin, h_fake):
        return torch.abs(h_origin - h_fake)

    def compute_diff_attention(self, video_name, image_features, fc):
        attention_history = {}
        attention_history[video_name] = []
        for i, frame_out in enumerate(image_features[:-1]):
            differences = []
            for j, frame_in in enumerate(image_features[i + 1:], start=i + 1):
                # calculate d1t, d2t, and d4t
                diff = int(j - i)
                if diff == 1 or diff == 2 or diff == 4:
                    with torch.no_grad():
                        difference = self.difference(frame_in, frame_out)
                        difference_ = fc(difference, diff)
                        differences.append(difference_.numpy())
            attention_history[video_name].append(np.sum(differences, axis=0))
        with torch.no_grad():
            last = fc(image_features[-1])
            attention_history[video_name].append(last.numpy())
        return attention_history[video_name]

    def train_places365_features(self, feat_dir):
        videos = listdir(feat_dir)

        attention_history = {}
        for idx, video in enumerate(videos):
            video_name = drop_file_extension(video.split('\\')[-1])
            video_path = path.join(feat_dir, video)
            image_features = torch.Tensor(open_pickle_file(video_path)).squeeze(1)
            attention_history[video_name] = self.compute_diff_attention(video_name, image_features, self.fc)

        return attention_history

    def train(self):
        if self.places365_features:
            places365_att = self.train_places365_features(self.places365_features)

        save_pickle_file(self.out_file, places365_att)


if __name__ == '__main__':
    diff_attention = DifferenceAttention(places365_features=DOWNSAMPLED_TVSUM_PLACES365_FEATURES , input_size=1024,
                                         out_file='tvsum_places365_attention')
    diff_attention.build()
    diff_attention.train()
    objects = []
    # with (open("summe_motion_features.pickle", "rb")) as openfile:
    #     while True:
    #         try:
    #             objects = pickle.load(openfile)
    #             print(objects)
    #         except EOFError:
    #             break
