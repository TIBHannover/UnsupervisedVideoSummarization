import json
import numpy as np
from collections import OrderedDict
from src.evaluation.summary_loader import load_processed_dataset
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd


sns.set()
sns.set_style("darkgrid")

videos = {}
x_axis = []
y_axis = []
original_splits = '../../results/TVSum/video_scores/original splits/'
non_overlapping_splits = '../../results/TVSum/video_scores/non overlapping splits/'
n_videos = 50
n_splits = 5

def read_scores(dir,n_splits,n_videos):
    df = pd.DataFrame(columns=['Videos', 'F1-scores'])
    for split in range(n_splits):
        path = dir + '/video_scores{}.txt'.format(split)
        print(path)
        with open(path, 'r') as infile:
            videos = json.load(infile)
            print(videos.keys())
            for key in videos.keys():
                # d = {'Videos': key, 'F1-scores': videos[key]}
                d = pd.Series({'Videos': key, 'F1-scores': videos[key]})
                df = df.append(d, ignore_index=True)

    df['Videos'] = df['Videos'].astype(int)
    series = df.groupby('Videos')['F1-scores'].mean()

    print(list(series.index.values))
    #print(len(list(series.index.values)))
    for i in range(1, n_videos + 1):
        if i not in list(series.index.values):
            x = pd.Series(0, index=[i])
            series = series.append(x)
    series = series.sort_index(ascending=True)
    #print(series)
    return series



original_videos = read_scores(original_splits,n_splits,n_videos)
non_overlapping_videos = read_scores(non_overlapping_splits,n_splits,n_videos)

diff= (non_overlapping_videos.values + original_videos.values)/2
print (diff)
df = pd.DataFrame({'Video Names':original_videos.index, 'F1-scores': diff, 'included':'', 'F1-score difference':'Values' })

plot=sns.scatterplot(x="Video Names", y="F1-scores",  markers=['o'], style= 'F1-score difference', data=df)
plt.xticks(np.arange(1, n_videos + 1))
plt.axhline(y=0, c='red', linestyle='dashed', label="horizontal")

#plt.yticks(np.arange(-90, 20, step=10))
plt.title("delta F1-score between original and non-overlapping video splits", fontsize= 15)
labels = ["Equivalent F1-score"]
handles, _ = plot.get_legend_handles_labels()

# Slice list to remove first handle
plt.legend(handles = handles[:], labels = labels)
plt.show()
