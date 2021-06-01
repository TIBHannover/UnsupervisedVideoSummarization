"""
This script maps the tvsum video numbers to their original names
"""
import pandas as pd

from src.evaluation.summary_loader import load_processed_dataset, load_tvsum_mat

PROCESSED_TVSUM = '../data/TVSum/processed/eccv16_dataset_tvsum_google_pool5.h5'
TVSUM_MAT = '../data/TVSUM/data/ydata-tvsum50.mat'

dataset = dataset = load_processed_dataset(PROCESSED_TVSUM)
df = pd.DataFrame.from_dict(dataset, orient='index')

results = load_tvsum_mat(TVSUM_MAT)

videos= {}
for idx, _ in enumerate(results):
    frames = results[idx]['user_anno'].shape[0]
    processed_name = df.loc[df['nframes']==int(frames)].index[0]
    videos[results[idx]['video'].split('.')[0]]= processed_name
print(videos)
print(len(videos.keys()))

for key in sorted(videos.keys()):
    print("%s: %s" % (key, videos[key]))