import re
import json
import os
import numpy as np
import subprocess
import cv2
import pickle
import h5py


def read_json(file_path):
    try:
        with open(file_path, 'r') as f:
            try:
                return json.load(f)
            except ValueError:
                raise ValueError('{} is not valid'.format(file_path))
    except IOError:
        raise IOError('{} does not exist.'.format(file_path))


def write_json(obj, file_path):
    make_directory(os.path.dirname(file_path))
    with open(file_path, 'w') as f:
        json.dump(obj, f)


def save_pickle_file(filename, data):
    print('Saving {} ...'.format(filename))
    with open('{}.pickle'.format(filename), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('{} saved'.format(filename))


def digits_in_string(text):
    non_digit = re.compile("\D")
    return int(non_digit.sub("", text))


def make_directory(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as err:
            raise

    return directory


def video2frames(video_path, fps, out_img_dir):
    subprocess.call(
        'ffmpeg -i {video} -vf fps={fps} {out_img_dir}/frame%06d.png > output.txt >> output.txt 2>&1 '.format(
            video=video_path,
            fps=fps,
            out_img_dir=out_img_dir
        ), shell=True)


def resize_image(img_path, width, height):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def sample_from_video_with_gt(video_frames, user_scores, duration, fps, n_samples=2):
    sampled_frames = []
    sampled_gt = []
    source_array = np.arange(fps)
    for i in range(duration):
        sorted_samples = make_sorted_sample(source_array, n_samples)
        start = i * fps
        end = (i + 1) * fps
        time_span = (start, end)
        sampled_frames.extend(select_frames_from_video(sorted_samples, video_frames, time_span))
        sampled_gt.extend(select_user_scores(sorted_samples, user_scores, time_span))
    return sampled_frames, sampled_gt



def downsample(video_frames,picks):
    frames= [video_frames[i] for i in picks]
    return frames

def select_frames_from_video(samples, video_frames, time_span):
    sampled_frames = []
    for idx in samples:
        sampled_frames.append(video_frames[time_span[0]:time_span[1]][idx - 1])
    return sampled_frames


def select_user_scores(samples, user_scores, time_span):
    sampled_gt = []
    for idx in samples:
        gt = [user_scores[time_span[0]:time_span[1], user][idx - 1] for user in range(user_scores.shape[1])]
        sampled_gt.append(np.asarray(gt))
    return sampled_gt


def make_sorted_sample(source_array, n_samples):
    sorted_samples = np.sort(np.random.choice(source_array, n_samples, replace=False))
    return sorted_samples


def drop_file_extension(file_name):
    if file_name is None:
        raise ValueError
    file_name = file_name.split('.')[:-1]
    return '.'.join(file_name)


def has_string(string, sub_string):
    if string is None or sub_string is None:
        raise ValueError
    string = str(string).lower()
    sub_string = str(sub_string).lower()

    return sub_string in string


def get_dir_in_path(dir):
    return [os.path.join(dir, o) for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]


def load_processed_dataset(type):
    dataset = h5py.File(type, 'r')
    return dataset

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
