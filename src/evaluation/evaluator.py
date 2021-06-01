"""
This script evaluates the results using kendalls tau and spearmans rho
"""
import argparse
import json
import os
from os import listdir

import numpy as np
from ProcessedDatasetEvaluator import ProcessedDatasetEvaluator
from summary_generator import upsample_summary
from summary_loader import load_processed_dataset

PROCESSED_SUMME = '../../data/SumMe/processed/eccv16_dataset_summe_google_pool5.h5'
PROCESSED_TVSUM = '../../data/TVSUM/processed/eccv16_dataset_tvsum_google_pool5.h5'







def average_results(video_scores):
    mean_scores = dict()
    for key in video_scores.keys():
        mean_scores[key] = np.asarray(video_scores[key]).mean(axis=0)

    return mean_scores


def resolve_video_names(json_file):
    with open(json_file) as f:
        data = json.loads(f.read())
        return list(data.keys())


def init_scores_dict(keys):
    all_scores = dict()
    all_scores = {el: [] for el in keys}  # init dict
    return all_scores


def evaluate_epoch_results(path, processed_dataset, metric,knapsack):
    results = listdir(path)
    #results.sort(key=lambda video: int(video[6:-5]))
    epoch0 = path + "/" + results[0]
    video_names = resolve_video_names(epoch0)
    video_scores = init_scores_dict(video_names)

    for idx, epoch in enumerate(results):
        with open(path + "/" + epoch) as f:
            data = json.loads(f.read())
            for video_name in video_names:
                n_frames = np.array(processed_dataset[video_name]['n_frames'])
                picks = np.array(processed_dataset[video_name]['picks'])
                change_points = np.array(processed_dataset[video_name]['change_points'])
                scores = np.array(data[video_name])
                # upsample the generated summary to equal the original
                upsampled_summary,_,_ = upsample_summary(scores, n_frames, picks, change_points,knapsack)
                evaluator = ProcessedDatasetEvaluator(upsampled_summary, processed_dataset, metric)
                ref_summaries = np.array(processed_dataset[video_name]['user_summary'])
                # evaluate
                video_scores[video_name].append(
                    evaluator.evaluate(upsampled_summary, ref_summaries))
    return video_scores


def arg_parser():
    #results = '../../results/SumMe/seed31_non_overlapping'
    #results = '../../results/SumMe/original_splits/seed31'

    #results = '../../results/TVSum/run1'
    #results = '../../results/TVSum/non_overlapping/run1'
    #results = '../../results/sum-gan-sl/TVSum/original-splits/seed20'
    #results = '../../results/sum-gan-sl/TVSum/non-overlapping-splits'

    #results = '../../results/sum-gan-aae/TVSum/non-overlapping-splits/run1'
    #results = '../../results/sum-gan-aae/TVSum/run1'



    #results = '../../results/csnet/SumMe/backup/original-splits/'
    #results = '../../results/csnet/SumMe/non-overlapping-splits/'

    #results = '../../results/csnet/TVSum/original-splits/'
    #results = '../../results/csnet/TVSum/non-overlapping-splits/'

    #results = '../../results/sum_ind/SumMe/fscores/original-splits'
    #results = '../../results/sum_ind/SumMe/fscores/non-overlapping-splits'

    results = '../../results/late_fusion/TVSum/non-overlapping-splits'
    #results = '../../results/intermediate_fusion/SumMe/original-splits'

    parser = argparse.ArgumentParser(description="Generate Summary")
    parser.add_argument("--results_path", type=str, default=results,
                        help="Scores path from a split")
    parser.add_argument("--dataset", type=str, default='tvsum',
                        help="two datasets are available: summe and tvsum")
    parser.add_argument("--metric", type=str, default='spearmanr',
                        help="metric spearmanr or kendalltau ")
    parser.add_argument("--knapsack", type=bool, default=False,
                        help="whether to apply knapsach and generate the end summary or not ")
    parser.add_argument("--binarize", type=bool, default=False)
    return parser


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    metric = args.metric
    results_path = args.results_path
    type = args.dataset
    knapsack=args.knapsack
    binarize= args.binarize

    # load dataset
    if type == 'summe':
        dataset = load_processed_dataset(PROCESSED_SUMME, type)
    else:
        dataset = load_processed_dataset(PROCESSED_TVSUM, type,binarize)

    n_splits = 5
    split_eval = []
    for idx in range(n_splits):
        split = os.path.join(results_path, 'split{}'.format(idx))
        video_scores = evaluate_epoch_results(split, dataset, metric,knapsack)
        # average the results over the total number of epochs
        # averaged = average_results(video_scores)
        split_eval.append(video_scores)
    print(split_eval)
    res = []
    for el in split_eval:
        res.append(list(el.values()))
    print(np.round(np.asarray(res).mean(), decimals=4))
