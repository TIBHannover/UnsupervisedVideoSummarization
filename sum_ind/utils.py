from __future__ import absolute_import
import os
import sys
import errno
import json
import os.path as osp
import numpy as np
import pickle
import torch


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def map_selected_frames_to_idxs(picks, seq):
    frames_idxs = dict()
    for idx, feat in zip(picks, seq):
        frames_idxs[idx] = feat
    return frames_idxs


def get_mean_frames_of_segments(segments):
    centers = []
    for key in segments:
        seg_length = len(segments[key])
        if seg_length > 0:
            if seg_length % 2 == 0:
                centers.append(segments[key][int(seg_length / 2) - 1:int(seg_length / 2) + 1][0])
            else:
                centers.append(np.median(segments[key]))

    centers = np.asarray(centers).astype(int)
    # centers = [seq[k] for k in seq if k in centers]
    return np.asarray(centers)


# returns the segments with idxs of frames in each segment
def get_frames_per_segments(n_frames_per_segments, picks):
    segments = dict()
    end = 0
    segment_id = 0
    for seg in n_frames_per_segments:
        start = end
        end += seg
        frame_idxs = [idx for idx, f in enumerate(picks) if f >= start and f < end]
        # delete segment if empty
        if len(frame_idxs) > 0:
            segments[segment_id] = frame_idxs
            segment_id = segment_id + 1

    return segments


def save_pickle_file(filename, data):
    print('Saving {} ...'.format(filename))
    with open('{}.pickle'.format(filename), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('{} saved'.format(filename))


def open_pickle_file(filename):
    with (open(filename, "rb")) as openfile:
        while True:
            try:
                return pickle.load(openfile)
            except EOFError:
                print(EOFError)
                break
