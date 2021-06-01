from scipy import stats
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd



def read_scores(dir, n_splits, correlation_scores):
    df = pd.DataFrame(columns=['Video Name', 'Average spearman\'s rank correlation between human annotators', 'F1-score'])
    for split in range(n_splits):
        path = dir + '/video_scores{}.txt'.format(split)
        print(path)
        with open(path, 'r') as infile:
            videos = json.load(infile)
            print(videos.keys())
            for key in videos.keys():
                video_name = 'video_' + key
                print(correlation_scores[video_name])
                d = {'Video Name': video_name, 'Average spearman\'s rank correlation between human annotators': correlation_scores[video_name], 'F1-score': videos[key]}
                df = df.append(d, ignore_index=True)
    print(df)
    return df


model = 'CSNet'
model_dir='csnet'
type = 'SumMe'
if type=='SumMe':
    path = 'summe_spearmanr.json'
elif type=='TVSum':
    path= 'tvsum_spearmanr.json'

with open(path, 'r') as infile:
     spearmans = json.load(infile)


summe_non_overlapping_splits = '../../results/{}/{}/video_scores/non overlapping/'.format(model_dir,type)

sns.set()
sns.set_style("darkgrid")

videos = {}
x_axis = []
y_axis = []
n_splits=5
correlation_scores= spearmans
df = read_scores(summe_non_overlapping_splits, n_splits, correlation_scores)

spearmanr, p_value = stats.spearmanr(df['Average spearman\'s rank correlation between human annotators'], df['F1-score'])
print('spearmanr: {}'.format(np.round(spearmanr,decimals=2)))

plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.labelsize"] = 15

plot = sns.regplot(x="Average spearman\'s rank correlation between human annotators", y="F1-score", data=df, ci=None,scatter_kws={'s':75})
plt.yticks(np.arange(start=0, stop=101, step=10))
#plt.title("{}: The spearman\'s rank correlation between F1-scores and average correlation of human annotators using non-overlapping {} video splits".format(model, type), fontsize=15)

labels = ["direction of the correlation","video names"]
handles, _ = plot.get_legend_handles_labels()

# Slice list to remove first handle
plt.legend(handles[:], labels = labels,fontsize=15)

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'], point['y'], str(point['val'][6:]), fontsize='15')


label_point(df['Average spearman\'s rank correlation between human annotators'], df['F1-score'], df['Video Name'],plt.gca())

plot.figure.savefig('{}-correlation-{}-.pdf'.format(model,type),format='pdf')

#plt.show()

