import os
import numpy as np
import pandas as pd
from BaseVideoClip import BaseVideoClip


class TVSumVideo(BaseVideoClip):
    GT_FILE = 'ydata-tvsum50-anno.tsv'

    def __init__(self, video_name, video_path, gt_base_dir):
        super().__init__(video_name, video_path, gt_base_dir)
        self.gt_path = os.path.join(gt_base_dir, self.GT_FILE)

    def get_gt(self):
        gt_df = pd.read_csv(self.gt_path, sep="\t", header=None, index_col=0)
        sub_gt = gt_df.loc[self.video_name]
        users_gt = []
        for i in range(len(sub_gt)):
            users_gt.append(sub_gt.iloc[i, -1].split(","))
        users_gt = np.array(users_gt)
        avg_gt = self.__avg_array(users_gt)
        return np.expand_dims(avg_gt, axis=1)

    def __avg_array(self, users_gt):
        users_gt = users_gt.astype(int)
        return np.rint(users_gt.mean(axis=0))

