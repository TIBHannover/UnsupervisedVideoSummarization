import argparse
import os

import numpy as np
from FeatureExtractor import FeatureExtractor
from PIL import Image
from SumMeVideo import SumMeVideo
from TVSumVideo import TVSumVideo
from VSUMMVideo import VSUMMVideo
from tqdm import tqdm
from utils import *

PROCESSED_SUMME = '../data/SumMe/processed/eccv16_dataset_summe_google_pool5.h5'
SUMME_MAPPED_VIDEO_NAMES = '../data/SumMe/mapped_video_names.json'
PROCESSED_TVSUM = '../data/TVSUM/processed/eccv16_dataset_tvsum_google_pool5.h5'
TVSUM_MAPPED_VIDEO_NAMES = '../data/TVSUM/mapped_video_names.json'

def create_video_obj(dataset, video_name, video_path, gt_dir):
    dataset = str(dataset).lower()
    if dataset == 'summe':
        return SumMeVideo(video_name, video_path, gt_dir)
    elif dataset == 'tvsum':
        return TVSumVideo(video_name, video_path, gt_dir)
    else:
        return VSUMMVideo(video_name, video_path, gt_dir)


def arg_parser():
    # ../data/SumMe/videos  ../data/SumMe/GT
    # ../ data / TVSum / video /  ../data/TVSum/data
    # ../data/VSUMM/new_database  ../data/VSUMM/newUserSummary
    parser = argparse.ArgumentParser(description='Extract Features')
    parser.add_argument('--dataset', default='summe', type=str, help='summe, tvsum or vsumm')
    parser.add_argument('--videos_dir', metavar='DIR', default='../data/SumMe/videos', help='path input videos')
    parser.add_argument('--gt', metavar='GT_Dir', default='../data/SumMe/GT', help='path ground truth')
    parser.add_argument('--fps', default=2, type=int, help='Frames per second for the extraction')
    parser.add_argument('--model_arch', default='resnet50',
                        help='pre-trained model architecture e.g. resnet50 or alexnet')
    return parser


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    # dataset, videos, GT, Sample rate, model architecture
    dataset, videos_dir, gt_dir, fps, model_arch = args.dataset, args.videos_dir, args.gt, args.fps, args.model_arch

    if dataset=='summe':
        processed_dataset=load_processed_dataset(PROCESSED_SUMME)
        mapped_video_names=read_json(SUMME_MAPPED_VIDEO_NAMES)
    else:
        load_processed_dataset(PROCESSED_TVSUM)
        mapped_video_names = read_json(TVSUM_MAPPED_VIDEO_NAMES)

    # define feature extractor model
    model = FeatureExtractor(model_arch)
    print(model)
    # stores all sampled data in array of dict
    features = dict()
    ground_truth = dict()

    # sort videos according to their number video1,video2,..
    all_videos = sorted(os.listdir(videos_dir), key=digits_in_string)

    # iterate over videos, sample frames and gt, and extract features
    for idx, video in enumerate(all_videos):
        features[idx] = []
        ground_truth[idx] = []
        # sample video an ground truth
        video_name = drop_file_extension(video)
        video_path = os.path.join(videos_dir, video)
        # create video according to dataset
        video_obj = create_video_obj(dataset, video_name, video_path, gt_dir)
        # get video frames
        print('getting frames for {}...'.format(video_name))
        video_frames = video_obj.get_frames()
        mapped_name=mapped_video_names[video_name]
        # sample video frames
        print('sampling from video frames...')
        sampled_frames = downsample(video_frames,processed_dataset[mapped_name]['picks'])
        # delete video object
        del video_obj
        print('frames retrieved')
        # iterate over sampled frames to extract their features using pretrained model
        print('Extracting features for {} ...'.format(video_name))
        for f in tqdm(range(len(sampled_frames))):
            # convert to PIL
            PIL_image = Image.fromarray(sampled_frames[f])
            frame = model.tranform(PIL_image)
            # extend dim
            frame = frame.view((1,) + frame.shape)
            # get features
            feat = model(frame)
            features[idx].append(feat.cpu().detach().numpy()[0])
        features[idx]=np.array(features[idx])
        save_pickle_file('features', features)
