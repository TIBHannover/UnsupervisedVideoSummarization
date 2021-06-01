import os
import numpy as np
from utils import digits_in_string, has_string, get_dir_in_path
from BaseVideoClip import BaseVideoClip


class VSUMMVideo(BaseVideoClip):
    def __init__(self, video_name, video_path, gt_base_dir):
        super().__init__(video_name, video_path, gt_base_dir)
        self.gt_path = os.path.join(gt_base_dir, video_name)

    def get_gt(self):
        vid_scores = []
        dirs = get_dir_in_path(self.gt_path)
        for i, user in enumerate(dirs):
            frame_idx = []
            for idx, summary in enumerate(os.listdir(user)):
                if has_string(summary, 'frame'):
                    frame_idx.append(digits_in_string(summary))  # Frame123--> 123
            vid_scores.append(frame_idx)
        binary_scores = self.bin_classify_user_score(vid_scores)

        return binary_scores


    def bin_classify_user_score(self, user_scores_list):
        result = []
        for frames_idx in list(user_scores_list):
            vid_frames = np.zeros(self.n_frames)
            np.put(vid_frames, frames_idx, [1])  # replace all selected frames with 1
            result.append(vid_frames)
        return np.asarray(result).T  # (n_frames,n_annotator)
