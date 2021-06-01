"""
This script evaluates the results of SUM-GAN-AAE, SUM-GAN-sl, and CSNET using F1-score metric
"""
import numpy as np
import os.path as osp
from os import listdir
import json
import argparse
import h5py
from summary_loader import load_processed_dataset
from summary_generator import generate_summary, upsample_summary
PROCESSED_SUMME = '../../data/SumMe/processed/eccv16_dataset_summe_google_pool5.h5'
PROCESSED_TVSUM = '../../data/TVSUM/processed/eccv16_dataset_tvsum_google_pool5.h5'

def evaluate_summary(predicted_summary, user_summary, eval_method):
    max_len = max(len(predicted_summary), user_summary.shape[1])
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[:len(predicted_summary)] = predicted_summary

    f_scores = []
    for user in range(user_summary.shape[0]):
        G[:user_summary.shape[1]] = user_summary[user]
        overlapped = S & G

        # Compute precision, recall, f-score
        precision = sum(overlapped) / sum(S)
        recall = sum(overlapped) / sum(G)
        if (precision + recall == 0):
            f_scores.append(0)
        else:
            f_scores.append(2 * precision * recall * 100 / (precision + recall))

    if eval_method == 'max':
        return max(f_scores)
    else:
        return sum(f_scores) / len(f_scores)




def resolve_video_names(json_file):
    with open(json_file) as f:
        data = json.loads(f.read())
        return list(data.keys())

def init_scores_dict(keys):
    all_scores = {el: {} for el in keys}  # init dict
    return all_scores

def evaluate_epoch_results(path, processed_dataset, eval_metric):
    results = listdir(path)
    results.sort(key=lambda video: int(video[6:-5]))
    epoch0 = path + "/" + results[0]
    video_names = resolve_video_names(epoch0)
    video_scores = init_scores_dict(video_names)

    for idx, epoch in enumerate(results):
        with open(path + "/" + epoch) as f:
            data = json.loads(f.read())
            for video_name in video_names:
                n_frames = np.array(processed_dataset[video_name]['n_frames'])
                positions = np.array(processed_dataset[video_name]['picks'])
                change_points = np.array(processed_dataset[video_name]['change_points'])
                scores = np.array(data[video_name])
                # upsample the generated summary to equal the original
                importance_scores,_,_ = upsample_summary(scores, n_frames, positions, change_points,False)
                generated_summmary, shot_lengths, selected_shot_idxs = upsample_summary(scores, n_frames, positions, change_points,True)

                ref_summaries = np.array(processed_dataset[video_name]['user_summary'])

                # evaluate
                video_scores[video_name]['importance_scores'] = importance_scores
                video_scores[video_name]['selected_shot_idxs'] = selected_shot_idxs
                video_scores[video_name]['shot_lengths'] = shot_lengths
                video_scores[video_name]['machine_summary'] = generated_summmary
                video_scores[video_name]['user_summary'] = ref_summaries
                video_scores[video_name]['fscore']= evaluate_summary(generated_summmary,ref_summaries, eval_metric)
    return video_scores


def arg_parser():
    #scores = '../../results/late-fusion/TVSum/non-overlapping-splits/'
    #results_path = '../../results/late-fusion/TVSum/fscores/non-overlapping-splits/'

    #scores = '../../results/intermediate-fusion/SumMe/non_overlapping/'
    #results_path = '../../results/intermediate-fusion/SumMe/fscores/non-overlapping-splits/'

    scores = '../../results/sum-gan-aae/SumMe/non-overlapping-splits/'
    results_path = '../../results/sum-gan-aae/SumMe/fscores/non-overlapping-splits/'

    #scores = '../../results/sum-gan-sl/SumMe/non-overlapping-splits/'
    #results_path = '../../results/sum-gan-sl/SumMe/fscores/non-overlapping-splits/'

    #scores = '../../results/sum-gan-aae/TVSum/non-overlapping-splits/run1'
    #results_path = '../../results/sum-gan-aae/TVSum/fscores/non-overlapping-splits/'

    parser = argparse.ArgumentParser(description="Generate Summary")
    parser.add_argument("--scores_path", type=str, default=scores,
                        help="Scores path from a split")
    parser.add_argument("--results_path", type=str, default=results_path,
                        help="where to save results")
    parser.add_argument("--metric", type=str, default='summe',
                        help="two datasets are available: summe and tvsum")
    return parser


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    metric = args.metric
    scores_path = args.scores_path
    results_path = args.results_path

    fms = []
    eval_metric = 'avg' if args.metric == 'tvsum' else 'max'
    if args.metric == 'tvsum':
        dataset = load_processed_dataset(PROCESSED_TVSUM,type='tvsum',binarize=True)

    else:
        dataset = h5py.File(PROCESSED_SUMME, 'r')



    n_splits=5
    split_eval = []
    for idx in range(n_splits):
        split = osp.join(scores_path, 'split{}'.format(idx))
        print(split)
        result = evaluate_epoch_results(split, dataset, eval_metric)
        print(result)
        h5_res = h5py.File(osp.join(results_path, '{}_split{}_result.h5'.format(args.metric, idx)), 'w')

        for key in result:
            h5_res.create_dataset(key + '/importance_scores', data=result[key]['importance_scores'])
            h5_res.create_dataset(key + '/selected_shot_idxs', data=result[key]['selected_shot_idxs'])
            h5_res.create_dataset(key + '/shot_lengths', data=result[key]['shot_lengths'])
            h5_res.create_dataset(key + '/machine_summary', data=result[key]['machine_summary'])
            h5_res.create_dataset(key + '/user_summary', data=result[key]['user_summary'])
            h5_res.create_dataset(key + '/fscore', data=result[key]['fscore'])

    h5_res.close()