import h5py
from matplotlib import pyplot as plt
import argparse
import os
import os.path as osp
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str,
                    default='log/non overlapping summe- SumInd (leaky-relu ) and (rep75+div25) layers 2/summe-split4/result.h5',
                    help="path to h5 file containing summarization results")
args = parser.parse_args()

parser.add_argument('-d', '--dataset', type=str, default='datasets/eccv16_dataset_summe_google_pool5.h5',
                    help="path to h5 dataaset")
args = parser.parse_args()

h5_res = h5py.File(args.path, 'r')
dataset = h5py.File(args.dataset, 'r')
keys = h5_res.keys()

def video_fscore_line_chart():
    for key in keys:
        score = h5_res[key]['score'][...]
        machine_summary = h5_res[key]['machine_summary'][...]
        gtscore = h5_res[key]['gtscore'][...]
        reference_summaries = np.asarray(dataset[key]['user_summary']).mean(axis=0)
        fm = h5_res[key]['fm'][()]
        n = len(reference_summaries)
        # plot score vs gtscore

        fig, ax = plt.subplots()
        line1, = ax.plot(range(n), reference_summaries, label='mean reference summaries')

        line2, = ax.plot(range(n), machine_summary, label='generated summary')

        ax.set_xlim(0, n)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0, .2, .4, .6, .8, 1])

        ax.set_title("generated summary for {},  with F-score {:.1%}".format(key, fm))
        ax.grid()
        ax.legend()

        fig.savefig(osp.join(osp.dirname(args.path), 'score_' + key + '.png'), bbox_inches='tight')
        plt.close()

        print("Done video {}. # frames {}.".format(key, len(machine_summary)))

    h5_res.close()

def get_all_fscores():
    n_splits = 5
    all_fscores = dict()
    for split in range(n_splits):
        path = 'log/non overlapping summe- SumInd (leaky-relu ) and (rep75+div25) layers 2/summe-split{}/result.h5'.format(split)
        split_results = h5py.File(path, 'r')
        keys = split_results.keys()
        for key in keys:
            fm = split_results[key]['fm'][()]
            all_fscores[key]=fm
    return all_fscores

def video_fscore_bar_chart():
    fscores = get_all_fscores()
    #print(fscores.sort)

if __name__ == '__main__':
    video_fscore_line_chart()
    #video_fscore_bar_chart()

