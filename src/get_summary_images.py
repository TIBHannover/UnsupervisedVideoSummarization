"""
This script gets the centers of the selected parts of a generated summary
"""
import argparse
import os
import h5py
from SumMeVideo import SumMeVideo
from TVSumVideo import TVSumVideo
from VSUMMVideo import VSUMMVideo
from utils import *
import numpy as np
from PIL import Image

SUMME_MAPPED_VIDEO_NAMES = '../data/SumMe/mapped_video_names.json'
TVSUM_MAPPED_VIDEO_NAMES = '../data/TVSUM/mapped_video_names.json'
results_path = '../results/csnet/SumMe/fscores/non-overlapping-splits/summe_split2_result.h5'
save_path = '../results/csnet/SumMe/fscores/non-overlapping-splits/'
dataset = 'summe'
videos_dir = '../data/SumMe/videos'


def arg_parser():
    parser = argparse.ArgumentParser(description='Extract Features')
    parser.add_argument('--dataset', default=dataset, type=str, help='summe, tvsum or vsumm')
    parser.add_argument('--videos_dir', metavar='DIR', default=videos_dir, help='path input videos')
    parser.add_argument('--results_path', type=str, default=results_path,
                        help="path to h5 file containing summarization results")
    parser.add_argument('--save_path', type=str, default=save_path)

    return parser


def create_video_obj(dataset, video_name, video_path, gt_dir):
    dataset = str(dataset).lower()
    if dataset == 'summe':
        return SumMeVideo(video_name, video_path, gt_dir)
    elif dataset == 'tvsum':
        return TVSumVideo(video_name, video_path, gt_dir)
    else:
        return VSUMMVideo(video_name, video_path, gt_dir)


def find_summary_segment_medians(summary):
    segment = []
    medians = []
    idxs = np.argwhere(summary == 1)
    prev = idxs[0]
    n = len(idxs)
    for i in idxs:
        if i - prev > 1:
            medians.append(int(np.round(np.median(segment))))
            segment = []
        segment.append(i)
        prev = i
        if i == idxs[n - 1] and len(segment) > 0:
            medians.append(int(np.round(np.median(segment))))
    return medians


def get_summary_images(h5_res, dataset, videos_dir,save_path):
    keys = h5_res.keys()
    for key in keys:
            machine_summary = h5_res[key]['machine_summary'][...]
            summary_medians = find_summary_segment_medians(machine_summary)
            video_path = os.path.join(videos_dir, get_original_video_name(dataset,key) + '.mp4')
            # create video according to dataset
            video_obj = create_video_obj(dataset, key, video_path, '')
            # get video frames
            print('getting frames for {}...'.format(key))
            video_frames = video_obj.get_frames()
            for i in summary_medians:
                PIL_image = Image.fromarray(video_frames[i])
                PIL_image.save(save_path+"{}_{}_frame_{}.png".format(key,dataset,i))
            # delete video object
            del video_obj

    h5_res.close()


def get_original_video_name(dataset, video):
    if dataset == 'summe':
        mapped_video_names= read_json(SUMME_MAPPED_VIDEO_NAMES)
    elif dataset == 'tvsum':
        mapped_video_names =read_json(TVSUM_MAPPED_VIDEO_NAMES)

    for original_name, _video in mapped_video_names.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if _video == video:
            return original_name


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    dataset, videos_dir,save_path= args.dataset, args.videos_dir, args.save_path
    if dataset == 'summe':
        mapped_video_names = read_json(SUMME_MAPPED_VIDEO_NAMES)
    else:
        mapped_video_names = read_json(TVSUM_MAPPED_VIDEO_NAMES)

    h5_res = h5py.File(args.results_path, 'r')

    get_summary_images(h5_res, dataset, videos_dir, save_path)
