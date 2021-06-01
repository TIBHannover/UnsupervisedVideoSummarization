from os import listdir
import json
import numpy as np
import h5py
from generate_summary import generate_summary
from evaluation_metrics import evaluate_summary
import yaml
from typing import Dict
import argparse

import pathlib


def init_scores_dict(keys):
    all_scores = {el: [] for el in keys}  # init dict
    return all_scores


def evaluate_split(split_path:str, dataset_path:str, eval_method:str):
    results = listdir(split_path)
    results.sort(key=lambda video: int(video[6:-5]))

    # for each epoch, read the results' file and compute the f_score
    f_score_epochs = []
    videos_scores = {}
    for idx, epoch in enumerate(results):
        print(epoch)
        all_scores = []
        with open(split_path + '/' + epoch) as f:
            data = json.loads(f.read())
            keys = list(data.keys())
            if idx == 0:
                video_numbers = [key[6:] for key in keys]
                videos_scores = init_scores_dict(video_numbers)

            for video_name in keys:
                scores = np.asarray(data[video_name])
                all_scores.append(scores)

        all_user_summary, all_shot_bound, all_nframes, all_positions = [], [], [], []
        with h5py.File(dataset_path, 'r') as hdf:
            for video_name in keys:
                video_index = video_name[6:]

                user_summary = np.array(hdf.get('video_' + video_index + '/user_summary'))
                sb = np.array(hdf.get('video_' + video_index + '/change_points'))
                n_frames = np.array(hdf.get('video_' + video_index + '/n_frames'))
                positions = np.array(hdf.get('video_' + video_index + '/picks'))

                all_user_summary.append(user_summary)
                all_shot_bound.append(sb)
                all_nframes.append(n_frames)
                all_positions.append(positions)

        all_summaries = generate_summary(all_shot_bound, all_scores, all_nframes, all_positions)

        all_f_scores = []
        # compare the resulting summary with the ground truth one, for each video
        for video_name, video_index in zip(keys, range(len(all_summaries))):
            summary = all_summaries[video_index]
            user_summary = all_user_summary[video_index]
            f_score = evaluate_summary(summary, user_summary, eval_method)
            all_f_scores.append(f_score)
            videos_scores[video_name[6:]].append(f_score)

        f_score_epochs.append(np.mean(all_f_scores))
        print("f_score: ", np.mean(all_f_scores))

    for key in videos_scores.keys():
        videos_scores[key] = np.round(np.mean(videos_scores[key]), decimals=2)

    return f_score_epochs


def compute_average_fscores(fscores:Dict):
    """
    method computes the final f1-score of the model
    :param fscores is a dict of split numbers as keys and fscores as values:
    :return:
    """
    all_fscores = []
    best_epochs = {}

    for split in range(len(fscores.keys())):
        split_result= fscores[split]
        print('split:{} , mean:{}'.format(split, np.mean(split_result)))
        all_fscores.append(np.mean(split_result))
        best_epochs['split{}'.format(split)] = np.argmax(np.asarray(split_result))
    print('f-scores.py for all splits: {}'.format(all_fscores))
    print('mean of  all fscores.py: {}'.format(np.mean(all_fscores)))
    print('median of  all fscores.py: {}'.format(np.median(all_fscores)))
    print('variance of  all fscores.py: {}'.format(np.var(all_fscores)))
    print('best epochs: {}'.format(best_epochs))



def arg_parser():
    parser = argparse.ArgumentParser(description='Extract Features')
    parser.add_argument('--dataset', default='summe', type=str, help='summe or tvsum')

    return parser




if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    config="config_summe.yaml"
    if args.dataset.strip().lower() == "tvsum":
        config = "config_tvsum.yaml"

    print(pathlib.Path(__file__).parent.absolute())
    config = yaml.load(open(pathlib.Path(__file__).parent / config), Loader=yaml.FullLoader)
    results = dict()
    for i, split in enumerate(config['splits']):
        split_path = '/'.join([config['path'],split])
        dataset_path = config['dataset_path']
        eval_method = config['eval_method']
        results[i] = evaluate_split(split_path, dataset_path, eval_method)
    compute_average_fscores(results)
    with open(config['path'] + '/' + config['fscores'], 'w') as outfile:
        json.dump(results, outfile)
