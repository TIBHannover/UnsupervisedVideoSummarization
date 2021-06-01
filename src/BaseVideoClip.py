from moviepy.editor import VideoFileClip
import abc
import os


class BaseVideoClip(VideoFileClip, abc.ABC):
    def __init__(self, video_name, video_path, gt_base_dir):
        self.video_clip = VideoFileClip(video_path)
        self.fps = int(self.video_clip.fps)
        self.duration = int(self.video_clip.duration)
        self.gt_path = os.path.join(gt_base_dir, video_name)
        self.video_name = video_name
        self.n_frames = self.video_clip.reader.infos['video_nframes']
    @abc.abstractmethod
    def get_gt(self):
        pass

    def get_frames(self):
        return list(self.video_clip.iter_frames(with_times=False))
