"""
This script maps the summe video numbers to their original names
"""
from os import listdir, path

import pandas as pd
from scipy.io import loadmat

from src.evaluation.summary_loader import load_processed_dataset

PROCESSED_SUMME = '../data/SumMe/processed/eccv16_dataset_summe_google_pool5.h5'
dataset = dataset = load_processed_dataset(PROCESSED_SUMME)
df = pd.DataFrame.from_dict(dataset, orient='index')
GT = '../data/SumMe/GT'
results = listdir(GT)

videos = {}
for idx, file in enumerate(results):
    video_path = path.join(GT, file)
    mat_file = loadmat(video_path)
    frames = mat_file['nFrames']
    processed_name = df.loc[df['nframes'] == int(frames)].index[0]
    videos[file.split('.')[0]] = processed_name
print(videos)
print(len(videos.keys()))
