# -*- coding: utf-8 -*-
import h5py
import numpy as np
import torch
import torch.nn as nn
import pickle
from os import listdir, path
from utils import drop_file_extension

PROCESSED_SUMME = 'data/SumMe/eccv16_dataset_summe_google_pool5.h5'
PROCESSED_TVSUM = 'data/TVSUM/eccv16_dataset_tvsum_google_pool5.h5'
DOWNSAMPLED_SUMME_FLOW_FEATURES = 'data/i3d_rgb_flow/saved_numpy_arrays/SumMe/I3D_features/FLOW/downsampled'
DOWNSAMPLED_SUMME_RGB_FEATURES = 'data/i3d_rgb_flow/saved_numpy_arrays/SumMe/I3D_features/RGB/downsampled'

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
    def __init__(self, dataset, flow_features=None, rgb_features=None, input_size=1024, out_file=''):
        """Class that Builds, Trains and Evaluates SUM-GAN model"""
        self.input_size = input_size
        self.dataset = h5py.File(dataset, 'r')
        self.flow_features = flow_features
        self.rgb_features = rgb_features
        self.out_file = out_file

    def build(self):

        # Build Modules
        self.fc = FC(
            self.input_size).to(device)
        self.fc_flow = FC(
            self.input_size).to(device)
        self.fc_rgb = FC(
            self.input_size).to(device)
        self.model = nn.ModuleList([
            self.fc, self.fc_flow, self.fc_rgb])

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

    def train_object_features(self):
        attention_history = {}
        for video_name in self.dataset.keys():
            attention_history[video_name] = []
            # [seq_len, 1024]
            image_features = torch.Tensor(self.dataset[video_name]['features'][...])
            attention_history[video_name] = self.compute_diff_attention(video_name, image_features, self.fc)

        return attention_history

    def train_motion_features(self, feat_dir, fc):
        videos = listdir(feat_dir)

        attention_history = {}
        for idx, video in enumerate(videos):
            video_name = drop_file_extension(video.split('\\')[-1])
            video_path = path.join(feat_dir, video)
            frame_features = np.load(video_path)
            attention_history[video_name] = []
            # [seq_len, 1024]
            image_features = torch.Tensor(frame_features)
            attention_history[video_name] = self.compute_diff_attention(video_name, image_features, fc)

        return attention_history

    def train(self):
        flow_features = {}
        # if self.dataset:
        #    object_features_att = self.train_object_features()
        if self.flow_features:
            flow_features['flow'] = self.train_motion_features(self.flow_features, self.fc_flow, )

        if self.rgb_features:
            flow_features['rgb'] = self.train_motion_features(self.rgb_features, self.fc_rgb)

        save_pickle_file(self.out_file, flow_features)


if __name__ == '__main__':
    # diff_attention = DifferenceAttention(dataset=PROCESSED_SUMME, flow_features=DOWNSAMPLED_SUMME_FLOW_FEATURES,
    #                                      rgb_features=DOWNSAMPLED_SUMME_RGB_FEATURES, input_size=1024,
    #                                      out_file='summe_motion_features')
    # diff_attention.build()
    # diff_attention.train()
    # objects = []
    with (open("summe_motion_features.pickle", "rb")) as openfile:
        while True:
            try:
                objects = pickle.load(openfile)
                print(objects)
            except EOFError:
                break
