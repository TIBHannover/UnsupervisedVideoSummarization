import os

import numpy as np
from BaseVideoClip import BaseVideoClip
from scipy.io import loadmat


class SumMeVideo(BaseVideoClip):
    def __init__(self, video_name, video_path, gt_base_dir):
        super().__init__(video_name, video_path, gt_base_dir)
        self.gt_path = os.path.join(gt_base_dir, video_name + '.mat')

    def get_gt(self):
        video_gt = loadmat(self.gt_path)
        video_gt = video_gt['user_score']  # (n_frames,n_annotator)
        return self.bin_classify_user_score(video_gt)

    def bin_classify_user_score(self, video_gt):
        user_scores = video_gt
        result = []
        for user in range(user_scores.shape[1]):
            # print(user_scores[:, user].shape)
            result.append([int(item > 0.5) for item in user_scores[:, user]])
        return np.asarray(result).T  # (n_frames,n_annotator)
