"""
This script evaluates the results of SUM-Ind using F1-score metric
"""
import h5py
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os
import os.path as osp
from os import listdir
import json
from fscore_evaluator import evaluate_summary
from summary_loader import load_processed_dataset
from summary_generator import generate_summary, upsample_summary

PROCESSED_SUMME = '../../data/SumMe/processed/eccv16_dataset_summe_google_pool5.h5'
PROCESSED_TVSUM = '../../data/TVSUM/processed/eccv16_dataset_tvsum_google_pool5.h5'
#metric = 'summe'
metric = 'tvsum'
results_dir = '../../results/sum_ind/TVSum/non-overlapping-splits'
#results_dir = '../../results/sum_ind/TVSum/original-splits'

if __name__ == "__main__":

    fms = []
    eval_metric = 'avg' if metric == 'tvsum' else 'max'
    eval_metric = 'avg'

    if metric == 'tvsum':
        dataset = load_processed_dataset(PROCESSED_TVSUM, type='tvsum', binarize=True)

    else:
        dataset = h5py.File(PROCESSED_SUMME, 'r')



    n_splits = 5
    split_eval = []
    all_fscores=[]
    for idx in range(n_splits):
        results_file = results_dir + "/split{}/result.h5".format(idx)
        h5_res = h5py.File(results_file, 'r')
        keys = h5_res.keys()
        json_file_path=results_dir + "/split{}".format(idx)
        videos={}

        for key in keys:
            videos[key]=[]
            scores = h5_res[key]['score'][...]
            videos[key] = scores.tolist()

            machine_summary = h5_res[key]['machine_summary'][...]
            gtscore = h5_res[key]['gtscore'][...]
            fm = h5_res[key]['fm'][()]

            n_frames = np.array(dataset[key]['n_frames'])
            positions = np.array(dataset[key]['picks'])
            change_points = np.array(dataset[key]['change_points'])

            importance_scores, _, _ = upsample_summary(scores, n_frames, positions, change_points, False)
            generated_summmary, shot_lengths, selected_shot_idxs = upsample_summary(scores, n_frames, positions,
                                                                                    change_points, True)
            ref_summaries = np.array(dataset[key]['user_summary'])

            fscore = evaluate_summary(generated_summmary, ref_summaries, eval_metric)
            all_fscores.append(fscore)
            print(key+ ': ' + str(fscore))

        afile = open(json_file_path + '/split{}.json'.format(idx), 'w')
        afile.write(json.dumps(videos))
        afile.close()
    print(np.round(np.mean(all_fscores),decimals=2))
    h5_res.close()
