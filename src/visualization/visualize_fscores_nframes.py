import json
import numpy as np
from collections import OrderedDict
from src.evaluation.summary_loader import load_processed_dataset
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

PROCESSED_SUMME = '../data/SumMe/processed/eccv16_dataset_summe_google_pool5.h5'
PROCESSED_TVSUM = '../../data/TVSUM/processed/eccv16_dataset_tvsum_google_pool5.h5'

sns.set()
sns.set_style("darkgrid")

videos = {}
x_axis = []
y_axis = []
#tvsum_original_splits = '../results/sum-gan-aae/TVSum/video_scores/original splits/'
tvsum_non_overlapping_splits = '../../results/sum-gan-aae/TVSum/video_scores/non overlapping/'
#tvsum_non_overlapping_splits = '../results/sum-gan-sl/TVSum/video_scores/non overlapping/'

#summe_original_splits = '../results/sum-gan-aae/SumMe/video_scores/original splits/'
summe_non_overlapping_splits = '../../results/sum-gan-aae/SumMe/video_scores/non overlapping/'
#summe_non_overlapping_splits = '../results/sum-gan-sl/SumMe/video_scores/non overlapping/'



def read_scores(dir, n_splits):
    df = pd.DataFrame(columns=['Number of Frames', 'F1-score', 'vid'])
    df.set_index('Number of Frames', inplace=True)
    for split in range(n_splits):
        path = dir + '/video_scores{}.txt'.format(split)
        print(path)
        with open(path, 'r') as infile:
            videos = json.load(infile)
            print(videos.keys())
            for key in videos.keys():
                nframes = dataset['video_' + key]['nframes']
                # d = {'Videos': key, 'F1-score': videos[key]}
                d = {'Number of Frames': nframes, 'F1-score': videos[key], 'vid': key}
                df = df.append(d, ignore_index=True)

    df = df.sort_index(ascending=True)

    df['Number of Frames'] = df['Number of Frames'].astype(int)
    df = df.groupby(['vid', 'Number of Frames'])['F1-score'].mean()
    df = df.sort_index(level=1)
    print(list(df.index.values))
    # print(len(list(series.index.values)))
    print(df)
    return df


type = 'tvsum'
model= 'SUM-GAN-sl'
n_splits = 5

step = 2500
if type == 'tvsum':
    dataset = load_processed_dataset(processed_dataset=PROCESSED_TVSUM)
    original_videos = read_scores(tvsum_non_overlapping_splits, n_splits)
    df = pd.DataFrame({'vid': original_videos.index.get_level_values(0),
                       'Number of Frames': original_videos.index.get_level_values(1),
                       'F1-score': original_videos.values})

    x_axis_stop = df['Number of Frames'].max() + step
    x_axis_start = df['Number of Frames'].min()



else:
    dataset = load_processed_dataset(processed_dataset=PROCESSED_SUMME)
    original_videos = read_scores(summe_non_overlapping_splits, n_splits)
    df = pd.DataFrame({'vid': original_videos.index.get_level_values(0),
                       'Number of Frames': original_videos.index.get_level_values(1),
                       'F1-score': original_videos.values})

    x_axis_stop = df['Number of Frames'].max() + step
    x_axis_start = df['Number of Frames'].min()

# diff= (non_overlapping_videos.values + original_videos.values)/2


plot = sns.scatterplot(x="Number of Frames", y="F1-score", data=df)
plt.xticks(np.arange(start=x_axis_start, stop=x_axis_stop, step=step))
plt.yticks(np.arange(start=0, stop=101, step=10))

plt.axhline(y=df['F1-score'].mean() + df['F1-score'].std(), c='blue', linestyle='dashed', label="horizontal")
plt.axhline(y=df['F1-score'].mean() - df['F1-score'].std(), c='blue', linestyle='dashed', label="horizontal")

# lowest 10 fscores
# print(df.loc[df['F1-score'].nsmallest(10).index])

sigma1_low = df['F1-score'].mean() - df['F1-score'].std()
print(df.loc[df['F1-score'] < sigma1_low])

# plt.yticks(np.arange(-90, 20, step=10))
plt.title("{}: F1-scores of non-overlapping {} video splits w.r.t video length".format(model,type), fontsize=15)
labels = ["Values within standard deviation of 1"]
handles, _ = plot.get_legend_handles_labels()

# Slicdfe list to remove first handle
plt.legend(handles=handles[:], labels=labels)
plt.show()
