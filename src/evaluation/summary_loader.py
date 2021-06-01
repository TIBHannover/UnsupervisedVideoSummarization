"""
This script loads the datasets
"""
import h5py
import hdf5storage
import numpy as np

PROCESSED_SUMME = '../../data/SumMe/processed/eccv16_dataset_summe_google_pool5.h5'
PROCESSED_TVSUM = '../../data/TVSUM/processed/eccv16_dataset_tvsum_google_pool5.h5'
TVSUM_MAT = '../../data/TVSUM/data/ydata-tvsum50.mat'


def load_tvsum_mat(filename=TVSUM_MAT):
    data = hdf5storage.loadmat(filename, variable_names=['tvsum50'])
    data = data['tvsum50'].ravel()

    data_list = []
    for item in data:
        video, _, _, _, _, user_anno, _ = item

        item_dict = {
            'video': video[0, 0],
            'user_anno': user_anno,
        }
        data_list.append(item_dict)

    return data_list


def binarize_scores(user_anno):
    median_vals = np.mean(user_anno, axis=0)
    for i, anno in enumerate(user_anno):
        for j in range(len(anno)):
            if anno[j] >= median_vals[j]:
                anno[j] = int(1)
            else:
                anno[j] = int(0)

    return user_anno


def load_processed_dataset(processed_dataset=PROCESSED_SUMME, type='summe', binarize=False):
    tvsum_mat = load_tvsum_mat(TVSUM_MAT) if type == 'tvsum' else None

    with h5py.File(processed_dataset, 'r') as hdf:
        data_list = dict()
        for video_name in hdf.keys():
            video_idx = video_name[6:]
            element = 'video_' + video_idx
            nframes = np.array(hdf.get(element + '/n_frames'))
            picks = np.array(hdf.get(element + '/picks'))
            change_points = np.array(hdf.get(element + '/change_points'))
            gt_score = np.array(hdf.get(element + '/gtsummary'))

            if type == 'tvsum':
                user_score = tvsum_mat[int(video_idx) - 1]['user_anno'].T
                if binarize:
                    print('binarize')
                    user_score = np.array(hdf.get(element + '/user_summary'))
            else:
                user_score = np.array(hdf.get(element + '/user_summary'))

            item_dict = {
                'n_frames': nframes,
                'user_summary': user_score,
                'picks': picks,
                'change_points': change_points,
                'gt_score': gt_score,
            }

            data_list[element] = item_dict

    hdf.close()
    return data_list
