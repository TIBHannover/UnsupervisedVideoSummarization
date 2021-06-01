# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json
from utils import open_pickle_file
import glob


class VideoData(Dataset):
    def __init__(self, mode, split_index, dataset):
        self.mode = mode
        self.name = dataset
        self.datasets = ['../data/SumMe/eccv16_dataset_summe_google_pool5.h5',
                         '../data/TVSum/eccv16_dataset_tvsum_google_pool5.h5']
        self.splits_filename = ['../data/splits/' + self.name + '_splits.json']
        self.splits = []
        self.split_index = split_index  # it represents the current split (varies from 0 to 4)
        temp = {}

        self.place365_datasets = ['../data/plc_365/place365_summe/',
                                '../data/plc_365/place365_tvsum/']


        if 'summe' in self.splits_filename[0]:
            self.filename = self.datasets[0]
            self.places_file_name = self.place365_datasets[0]
        elif 'tvsum' in self.splits_filename[0]:
            self.filename = self.datasets[1]
            self.places_file_name = self.place365_datasets[1]
        # read dataset files
        self.video_data = h5py.File(self.filename, 'r')


        with open(self.splits_filename[0]) as f:
            data = json.loads(f.read())
            for split in data:
                temp['train_keys'] = split['train_keys']
                temp['test_keys'] = split['test_keys']
                self.splits.append(temp.copy())

    def __len__(self):
        self.len = len(self.splits[0][self.mode + '_keys'])
        return self.len

    # In "train" mode it returns the features; in "test" mode it returns also the video_name
    def __getitem__(self, index):
        video_name = self.splits[self.split_index][self.mode + '_keys'][index]
        # processed dataset
        frame_features = torch.Tensor(np.array(self.video_data[video_name + '/features']))
        # flow features
        places365_features = self.places_file_name + video_name + '.p'
        places365_features = torch.Tensor(open_pickle_file(places365_features)).squeeze(1)


        if self.mode == 'test':
            return frame_features, video_name, places365_features
        else:
            return frame_features, video_name, places365_features


def get_loader(mode, split_index,dataset='summe', places365_features= False):
    if mode.lower() == 'train':
        vd = VideoData(mode, split_index, dataset, places365_features)
        return DataLoader(vd, batch_size=1)
    else:
        return VideoData(mode, split_index, dataset, places365_features)


def get_difference_attention(dataset):
    if dataset == 'summe':
        return open_pickle_file('summe_diff_attention.pickle'), open_pickle_file('summe_places365_attention.pickle')
    elif dataset == 'tvsum':
        return open_pickle_file('tvsum_diff_attention.pickle')
    else:
        return None


if __name__ == '__main__':
    pass
