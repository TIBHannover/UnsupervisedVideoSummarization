import h5py
from matplotlib import pyplot as plt
import argparse
import os
import os.path as osp
import numpy as np
from os import listdir
import seaborn as sns
import pandas as pd



sns.set()
sns.set_style("darkgrid")
sns.set_theme(style="darkgrid")

results_path = '../../results/csnet/SumMe/fscores/non-overlapping-splits/split0_result.h5'
save_path = '../../results/csnet/SumMe/fscores/non-overlapping-splits/'

#results_path = '../../results/sum-gan-aae/TVSum/fscores/non-overlapping-splits/tvsum_split0_result.h5'
#save_path = '../../results/sum-gan-aae/TVSum/fscores/non-overlapping-splits/'

PROCESSED_SUMME = '../../data/SumMe/processed/eccv16_dataset_summe_google_pool5.h5'
PROCESSED_TVSUM = '../../data/TVSUM/processed/eccv16_dataset_tvsum_google_pool5.h5'

parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str, default=results_path,
                    help="path to h5 file containing summarization results")
parser.add_argument('--save_path', type=str, default=save_path)

args = parser.parse_args()

parser.add_argument('-d', '--dataset', type=str, default='summe')


def video_fscore_line_chart(h5_res, save_path):
    step=2500
    keys = h5_res.keys()
    for key in keys:
        machine_scores = h5_res[key]['machine_scores'][...]


        reference_summaries = np.asarray(h5_res[key]['user_summary']).mean(axis=0)
        print(reference_summaries)
        fm = h5_res[key]['fscore'][()]
        n = reference_summaries.shape[0]
        # plot score vs gtscore

        #fig, ax = plt.subplots()

        df = pd.DataFrame({
            'Frame Index': np.arange(n),
            'machine scores': machine_scores,
            'training ground truth summary': reference_summaries,

        })
        plot = sns.lineplot(x='Frame Index', y='value', hue='variable',
                     data=pd.melt(df, ['Frame Index']))

       # line1, = ax.plot(range(n), reference_summaries, label='mean reference summaries')

        #line2, = ax.plot(range(n), machine_summary, label='generated summary')
        #plt.xticks(np.arange(start=x_axis_start, stop=x_axis_stop, step=step))
        #plt.yticks(np.arange(start=0, stop=101, step=10))
        #plot.set_xlim(0, n+1)
        plt.xlim(0,n+1)
        #plot.set(xlim=(0, n))
        plot.set(ylim=(0, 1))
        #plot.set_ylim(0, 1.1)
        #plot.set_yticks([0, .2, .4, .6, .8, 1])
        plot.set_ylabel("Frame Importance Score")
        plot.set_title("generated summary for {},  with F-score {:.2f}%".format(key, fm))
        #ax.grid()
        #ax.legend()
        plt.show()

        plot.figure.savefig(osp.join(osp.dirname(save_path), 'machine_score_' + key + '.png'))
        plt.close()

        print("Done video {}. # frames {}.".format(key, len(machine_scores)))

    h5_res.close()



if __name__ == '__main__':
    args = parser.parse_args()
    if args.dataset == 'tvsum':
        dataset = h5py.File(PROCESSED_TVSUM, 'r')
    else:
        dataset = h5py.File(PROCESSED_SUMME, 'r')

    h5_res = h5py.File(args.results_path, 'r')

    video_fscore_line_chart(h5_res, args.save_path)
    # video_fscore_bar_chart()
