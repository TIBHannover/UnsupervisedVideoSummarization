# -*- coding: utf-8 -*-
import h5py
import numpy as np
import torch
import torch.nn as nn
import pickle

PROCESSED_SUMME = 'data/SumMe/eccv16_dataset_summe_google_pool5.h5'
PROCESSED_TVSUM = 'data/TVSUM/eccv16_dataset_tvsum_google_pool5.h5'

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
    def __init__(self, dataset, input_size=1024, out_file=''):
        """Class that Builds, Trains and Evaluates SUM-GAN model"""
        self.input_size = input_size
        self.dataset = h5py.File(dataset, 'r')
        self.out_file= out_file

    def build(self):

        # Build Modules
        self.fc = FC(
            self.input_size).to(device)
        self.model = nn.ModuleList([
            self.fc])

        self.model.train()
        print(self.model)

    def difference(self, h_origin, h_fake):
        return torch.abs(h_origin - h_fake)

    def train(self):
        step = 0
        attention_history = {}

        for video_name in self.dataset.keys():
            attention_history[video_name] = []
            # [seq_len, 1024]
            image_features = torch.Tensor(self.dataset[video_name]['features'][...])
            for i, frame_out in enumerate(image_features[:-1]):
                differences = []
                for j, frame_in in enumerate(image_features[i + 1:], start=i + 1):
                    # calculate d1t, d2t, and d4t
                    diff = int(j - i)
                    if diff == 1 or diff == 2 or diff == 4:
                        with torch.no_grad():
                            difference = self.difference(frame_in, frame_out)
                            difference_ = self.fc(difference, diff)
                            differences.append(difference_.numpy())
                # print(len(differences))
                # sum the differences
                attention_history[video_name].append(np.sum(differences, axis=0))
            with torch.no_grad():
                last = self.fc(image_features[-1])
                attention_history[video_name].append(last.numpy())
            save_pickle_file(self.out_file, attention_history)


if __name__ == '__main__':
    diff_attention = DifferenceAttention(dataset=PROCESSED_SUMME, input_size=1024, out_file = 'test')
    diff_attention.build()
    diff_attention.train()
    objects = []
    # with (open("summe_diff_attention.pickle", "rb")) as openfile:
    #     while True:
    #         try:
    #             objects = pickle.load(openfile)
    #         except EOFError:
    #             break
