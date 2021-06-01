import h5py
from matplotlib import pyplot as plt
import argparse
import os
import os.path as osp
import numpy as np
from os import listdir
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches



sns.set()
sns.set_style("darkgrid")


save_path = '../../results/csnet/SumMe/fscores/non-overlapping-splits/'
results_path = '../../results/csnet/SumMe/fscores/non-overlapping-splits/summe_split4_result.h5'

#save_path = '../../results/intermediate-fusion/SumMe/fscores/non-overlapping-splits/'
#results_path = '../../results/intermediate-fusion/SumMe/fscores/non-overlapping-splits/summe_split4_result.h5'


#results_path = '../../results/csnet/TVSum/fscores/non-overlapping-splits/tvsum_split1_result.h5'
#save_path = '../../results/csnet/TVSum/fscores/non-overlapping-splits/'

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
        print(key)
        machine_summary = h5_res[key]['machine_summary'][...]

        reference_summaries = np.asarray(h5_res[key]['user_summary']).mean(axis=0)

        fm = h5_res[key]['fscore'][()]
        print(fm)
        n = reference_summaries.shape[0]
        df = pd.DataFrame({
            'Frame Index': np.arange(n),
            'machine summary': machine_summary,
            'training ground truth summary': reference_summaries,

        })


        # set color palette
        palette=sns.color_palette("Set2")
        _palette = sns.set_palette(palette)

        # plot dimensions
        plt.rcParams["figure.figsize"] = (3.3, 1.0)
        plt.rc('xtick', labelsize=8)
        plt.rc('ytick', labelsize=8)
        plt.rc('axes', labelsize=8)
        plt.rcParams['xtick.major.pad'] = '0'
        plt.rcParams['ytick.major.pad'] = '0'

        fig, ax = plt.subplots()
        ax.plot(range(n), reference_summaries, label='mean reference summaries')
        ax.plot(range(n), machine_summary, label='generated summary')
        ax.set_xlim(0,n+1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Importance Score")
        ax.set_xlabel("Frame Index")


        #plot.set_title("CSNet: Predicted summary vs. Ground truth summary for {}, with F1-score {:.2f}%".format(key, fm), fontsize=20)
        gt_patch = mpatches.Patch( label='ground truth',color=palette[0])
        tpositive_patch = mpatches.Patch( label='predicted', color=palette[1])
        fpositives_patch = mpatches.Patch( label='false positives', color=palette[2])
        plt.margins(0, 0)

        #plt.legend(handles=[gt_patch,tpositive_patch,fpositives_patch],loc='upper right' ,fontsize=20)
        plt.legend(handles=[gt_patch, tpositive_patch], loc='upper right', fontsize=6)

        #plt.show()
        ax.figure.savefig(osp.join(osp.dirname(save_path), 'summary_' + key + '.pdf'),  bbox_inches = 'tight',
    pad_inches = 0.0)
        plt.close()

        #print("Done video {}. # frames {}.".format(key, len(machine_summary)))

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
