import json
import numpy as np
from collections import OrderedDict
from src.evaluation.summary_loader import load_processed_dataset
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.patches as mpatches


sns.set()
sns.set_style("darkgrid")

n_videos = 25
videos = {}

n_splits = 5
x_axis = []
y_axis = []

model='csnet'
model_name='CSNet'
dataset_type='SumMe'
df = pd.DataFrame(columns=['Video Names', 'F1-score', 'Split Type'])

# cross validation splits
for split in range(n_splits):
    path = '../../results/{}/{}/video_scores/original splits/video_scores{}.txt'.format(model,dataset_type,split)
    print(path)
    with open(path, 'r') as infile:

        videos = json.load(infile)
        print(videos.keys())
        for key in videos.keys():
            # d = {'Videos': key, 'F1-score': videos[key]}
            d = pd.Series({'Video Names': key, 'F1-score': videos[key], 'Split Type': 'original splits'})
            df = df.append(d, ignore_index=True)

# cross validation splits
for split in range(n_splits):
    path = '../../results/{}/{}/video_scores/non overlapping/video_scores{}.txt'.format(model,dataset_type,split)
    print(path)
    with open(path, 'r') as infile:

        videos = json.load(infile)
        print(videos.keys())
        for key in videos.keys():
            # d = {'Videos': key, 'F1-score': videos[key]}
            d = pd.Series({'Video Names': key, 'F1-score': videos[key], 'Split Type': 'non-overlapping splits'})
            df = df.append(d, ignore_index=True)

df['Video Names'] = df['Video Names'].astype(int)
df = df.sort_values(by=['Video Names'])

print(df)
palette = sns.color_palette()
_palette = sns.set_palette(palette)

custom_palette = {}
for q in set(df['Split Type']):
    if 'original' in q:
        custom_palette[q] = palette[1]
    else:
        custom_palette[q] = palette[0]
g=sns.catplot(x="Video Names", y="F1-score", hue="Split Type", data=df, kind="bar", orient='v', height=6, aspect=20/7, palette=custom_palette, ci=None, legend_out=False)
plt.title("Performance of {} on {} videos w.r.t split type".format(model_name,dataset_type), fontsize=15)
plt.yticks(np.arange(start=0, stop=101, step=10))
g.despine(left=False)

plt.legend(loc='upper right')
#plt.show()
g.savefig('{}-{}-barchart.png'.format(model,dataset_type),format='png',dpi=300)


# in figure config use right 85 and top


