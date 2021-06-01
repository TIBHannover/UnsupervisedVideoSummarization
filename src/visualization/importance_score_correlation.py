""""

Code from: https://github.com/mayu-ot/rethinking-evs

"""
import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc
from src.evaluation.summary_loader import load_processed_dataset,load_tvsum_mat
import argparse
import pandas as pd

sns.set()


def accum_eval(pred, gt):
    total = gt.mean(axis=0).sum()
    x = np.argsort(pred)[::-1]

    y = [0]
    for i in range(x.size):
        cur_score = y[-1]
        y.append(cur_score + gt[:, x[i]].mean())
    y = np.asarray(y[1:]) / total
    return y


def best_curve(gt):
    total = gt.mean(axis=0).sum()
    x = np.argsort(gt.mean(axis=0))[::-1]

    y = [0]
    for i in range(x.size):
        cur_score = y[-1]
        y.append(cur_score + gt[:, x[i]].mean())

    y = np.asarray(y[1:]) / total

    return y


def worst_curve(gt):
    total = gt.mean(axis=0).sum()
    x = np.argsort(gt.mean(axis=0))

    y = [0]
    for i in range(x.size):
        cur_score = y[-1]
        y.append(cur_score + gt[:, x[i]].mean())
    y = np.asarray(y[1:]) / total
    return y


def plot_user_scores(user_anno,color='lightcoral'):
    user_anno = (user_anno - 1.) / 4.
    N = len(user_anno)

    plt.figure(figsize=(5, 5))

    # upper-bound
    best_y = best_curve(user_anno)
    best_auc = auc(np.linspace(0, 1, best_y.size), best_y)

    # lower-bound
    worst_y = worst_curve(user_anno)
    worst_auc = auc(np.linspace(0, 1, worst_y.size), worst_y)

    plt.fill_between(range(len(best_y)), worst_y, best_y, color='lightblue', alpha=.5)

    mean_auc = 0
    for i in range(N):
        pred = user_anno[i]
        y = accum_eval(pred, user_anno[list(range(i)) + list(range(i + 1, N))])
        mean_auc += auc(np.linspace(0, 1, y.size), y)
        p0 = plt.plot(y, color=color, alpha=.5)

    return mean_auc, best_auc, worst_auc, p0

def plot_model_scores(user_anno,color='lightcoral'):
    user_anno = (user_anno - 1.) / 4.
    plt.figure(figsize=(5, 5))
    p0 = plt.plot(user_anno, color=color, alpha=.5)

    mean_auc = 0
    for i in range(1):
        pred = user_anno[i]
        y = accum_eval(pred, user_anno[list(range(i)) + list(range(i + 1, 1))])
        mean_auc += auc(np.linspace(0, 1, y.size), y)
        p0 = plt.plot(y, color=color, alpha=.5)

    return p0


def plot_curve_rlvsumm(h5_res, save_path):

    human_auc_summary = []
    random_auc_summary = []
    rel_human_auc = []
    rel_random_auc = []

    keys = h5_res.keys()
    for key in keys:
        user_summary = h5_res[key]['user_summary'][...]
        importance_scores = h5_res[key]['importance_scores'][...]
        n_fr = user_summary.shape[1]
        N = len(user_summary)

        human_mean_auc, best_auc, worst_auc, p0 = plot_user_scores(user_summary)
        p2 = plot_model_scores(np.expand_dims(importance_scores,axis=1),color='blue')

        human_auc_summary.append(human_mean_auc / N)

        rel_human_auc.append((human_auc_summary[-1] - worst_auc) / (best_auc - worst_auc) * 100)

        # plot curve by random scoring
        pred = np.random.random((n_fr))
        y = accum_eval(pred, user_summary)
        p1 = plt.plot(y, color='k', linestyle='--')

        random_auc = auc(np.linspace(0, 1, y.size), y)
        random_auc_summary.append(random_auc)

        rel_random_auc.append((random_auc - worst_auc) / (best_auc - worst_auc) * 100)

        plt.legend((p0[0], p1[0],p2[0]), ('Humans', 'Random','csnet'))
        plt.title(key)
        plt.show()
        plt.close('all')

    print('random:', sum(random_auc_summary) / len(random_auc_summary), sum(rel_random_auc) / len(rel_random_auc), '\n',
          'human:', sum(human_auc_summary) / len(human_auc_summary), sum(rel_human_auc) / len(rel_human_auc))


results_path = '../../results/csnet/TVSum/fscores/non-overlapping-splits/tvsum_split0_result.h5'
save_path = '../../results/csnet/TVSum/fscores/non-overlapping-splits/'

#results_path = '../../results/sum-gan-aae/TVSum/fscores/non-overlapping-splits/tvsum_split1_result.h5'
#save_path = '../../results/sum-gan-aae/TVSum/fscores/non-overlapping-splits/'

PROCESSED_SUMME = '../../data/SumMe/processed/eccv16_dataset_summe_google_pool5.h5'
PROCESSED_TVSUM = '../../data/TVSUM/processed/eccv16_dataset_tvsum_google_pool5.h5'

parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str, default=results_path,
                    help="path to h5 file containing summarization results")
parser.add_argument('--save_path', type=str, default=save_path)

args = parser.parse_args()

parser.add_argument('-d', '--dataset', type=str, default='tvsum')



if __name__ == '__main__':
    args = parser.parse_args()
    if args.dataset == 'tvsum':
        dataset = h5py.File(PROCESSED_TVSUM, 'r')
    else:
        dataset = h5py.File(PROCESSED_SUMME, 'r')

    h5_res = h5py.File(args.results_path, 'r')


    sns.set_context('notebook', font_scale=1.3)
    plot_curve_rlvsumm(h5_res, args.save_path)