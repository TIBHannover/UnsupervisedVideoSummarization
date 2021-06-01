"""

This script computes the correlation between each two human annotators for a given dataset
"""

from summary_loader import load_processed_dataset
from scipy.stats import kendalltau, spearmanr
from scipy.stats import rankdata
import numpy as np
import collections

PROCESSED_SUMME = '../../data/SumMe/processed/eccv16_dataset_summe_google_pool5.h5'
PROCESSED_TVSUM = '../../data/TVSum/processed/eccv16_dataset_tvsum_google_pool5.h5'

dataset = load_processed_dataset(PROCESSED_TVSUM, type='tvsum')


def get_rc_func(metric):
    if metric == 'kendalltau':
        f = lambda x, y: kendalltau(rankdata(-x), rankdata(-y))
    elif metric == 'spearmanr':
        f = lambda x, y: spearmanr(x, y)
    else:
        raise RuntimeError
    return f

class HumanEvaluator():
    def __init__(self, metric):
        self.rc_func = get_rc_func(metric)

    def __call__(self):
        res = []
        for video_name in dataset.keys():
            user_anno = dataset[video_name]['user_score']
            N = user_anno.shape[1]

            max_rc = []
            min_rc = []
            avr_rc = []
            rc = []
            for i, x in enumerate(user_anno):
                R = [self.rc_func(x, user_anno[j])[0] for j in range(len(user_anno)) if j != i]

                max_rc.append(max(R))
                min_rc.append(min(R))
                avr_rc.append(np.mean(R))
                rc += R

            res.append({'video': dataset[video_name],
                        'video_name': video_name,
                        'mean': np.mean(avr_rc),
                        'min': np.mean(min_rc),
                        'max': np.mean(max_rc),
                        'cc': np.asarray(rc)
                        })
        return res


metric = 'spearmanr'
human_res = HumanEvaluator(metric)()
#print(human_res)
res=dict()
mean_arr = np.asarray([x['mean'] for x in human_res])

for x in human_res:
    res[x['video_name']]=x['mean']
print('human' + ': mean %.3f' % (np.mean(mean_arr)))
print(res)

#print(np.sort(mean_arr,axis=1))
print({k: v for k, v in sorted(res.items(), key=lambda item: item[1])})
