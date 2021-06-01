import json
import numpy as np
from collections import OrderedDict
from src.evaluation.summary_loader import load_processed_dataset
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

sns.set()
sns.set_style("darkgrid")

n_videos = 50
videos = {}

n_splits = 5
x_axis = []
y_axis = []

df = pd.DataFrame(columns=['Videos', 'F1-scores', 'Split Type'])

# original splits
for split in range(n_splits):
    path = '../results/TVSum/video_scores/original splits/video_scores{}.txt'.format(split)
    print(path)
    with open(path, 'r') as infile:

        videos = json.load(infile)
        print(videos.keys())
        for key in videos.keys():
            # d = {'Videos': key, 'F1-scores': videos[key]}
            d = pd.Series({'Videos': key, 'F1-scores': videos[key], 'Split Type': 'Original splits'})
            df = df.append(d, ignore_index=True)

# non-overlapping splits
for split in range(n_splits):
    path = '../results/TVSum/video_scores/non overlapping splits/video_scores{}.txt'.format(split)
    print(path)
    with open(path, 'r') as infile:

        videos = json.load(infile)
        print(videos.keys())
        for key in videos.keys():
            # d = {'Videos': key, 'F1-scores': videos[key]}
            d = pd.Series({'Videos': key, 'F1-scores': videos[key], 'Split Type': 'non-overlapping splits'})
            df = df.append(d, ignore_index=True)

# y_axis = list(videos.values())
# x_axis = list(videos.keys())

# print(list(x_axis))
# d = {'Videos': x_axis, 'F1-scores': y_axis, 'u':[True,True,True,True,True]}

df['Videos'] = df['Videos'].astype(int)
df = df.sort_values(by=['Videos'])

print(df)

sns.relplot(x="Videos", y="F1-scores", dashes=True, style='Split Type', hue='Split Type', markers=True, kind="line",
            data=df)
plt.xticks(np.arange(1, n_videos + 1))
plt.show()

# print(x_axis)
# print(y_axis)
