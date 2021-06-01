"""
This script computes and visualizes the usage percentage of videos used in a dataset i.e. the amount of videos included in test data
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns


sns.set()
summe = []
tvsum = []

sns.set(font_scale = 1.2)
plt.rcParams['axes.titlepad'] = 20
plt.rcParams["axes.labelsize"] = 20


for i in range(25):
    summe.append('video_{}'.format(i + 1))
for i in range(50):
    tvsum.append('video_{}'.format(i + 1))

all_fscores = []
best_epochs = {}
#path = '../../data/splits/summe_splits.json'
path = '../../data/splits/tvsum_splits.json'

test = []
print(path)
with open(path, 'r') as infile:
    data = json.load(infile)
    for split in data:
        test.append(split['test_keys'])
flattened_list = list(itertools.chain(*test))
unique_items = list(dict.fromkeys(flattened_list))

all = dict()
for item in flattened_list: all[item] = 0

for item in flattened_list:
    all[item] += 1

print(all)
print(unique_items)
print(len(unique_items))

cmap = plt.get_cmap("tab20c")
inner_colors = outer_colors = cmap(np.arange(3) * 2)

fig1, ax1 = plt.subplots()


def func(pct, allvals):
    print(pct)
    absolute = int(np.round(pct / 100. * np.sum(allvals)))
    return "{:.1f}%\n({:d} video)".format(pct, absolute)


data = [34.0, 16]
#data = [18, 7]

ax1.pie(data, autopct=lambda pct: func(pct, data), colors=inner_colors,
        shadow=False, startangle=90)

ax1.set_title('TVSum Video Usage\n #videos: 50')
#ax1.set_title('SumMe Video Usage\n #videos: 25')
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.legend(['used videos', 'unused videos'],
           loc="center left",
           bbox_to_anchor=(.65, 0, 0.5, .75),
           fontsize=15)

#plt.show()
ax1.figure.savefig('tvsum_usage.pdf', bbox_inches='tight',
                    pad_inches=0.0)
