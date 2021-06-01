# -*- coding: utf-8 -*-
"""
Created on Mon May 28 13:07:24 2018

@author: ziyad
"""

import glob
import cv2
import numpy as np
from utils import *

PROCESSED_SUMME = '../data/SumMe/processed/eccv16_dataset_summe_google_pool5.h5'
SUMME_MAPPED_VIDEO_NAMES = '../data/SumMe/mapped_video_names.json'
PROCESSED_TVSUM = '../data/TVSUM/processed/eccv16_dataset_tvsum_google_pool5.h5'
TVSUM_MAPPED_VIDEO_NAMES = '../data/TVSUM/mapped_video_names.json'


class ReadFileToNumpy(object):

    def vid2npy3(self, fileName):
        cap = cv2.VideoCapture(fileName)
        #        videoFrameCount = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        videoFrameCount = int(cap.get(7))

        frameCount = videoFrameCount
        # frameWidth = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        # frameHeight = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        frameHeight = 224
        frameWidth = 224
        dimensions = 3
        buf = np.empty((frameCount, frameHeight, frameWidth, dimensions), np.dtype('uint8'))
        fc = 0
        ret = True
        while (fc < frameCount and ret):
            ret, frame = cap.read()
            try:
                #            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resize = cv2.resize(frame, (frameHeight, frameWidth))

                buf[fc] = resize
                fc += 1
            except:
                continue

        cap.release()
        return buf, frameCount


rftn = ReadFileToNumpy()

# fileNames = glob.glob('../../Webscope_I4/ydata-tvsum50-v1_1/video/*.mp4')
fileNames = glob.glob('./../data/SumMe/videos/*.mp4')


# def RGB_numpy_arrays():
def RGB_numpy_arrays(fileNames):
    for i in range(len(fileNames)):
        videoFile = fileNames[i]
        #        videoMatFile = matFileNames[i]

        video, frameCount = rftn.vid2npy3(videoFile)
        # targets = extract.getTargets(videoMatFile, 4480)
        new_fileName = videoFile.split('/')[-1].split('.')[0]
        print(new_fileName)
        # save numpy arrays
        #        np.save('../../saved_numpy_arrays/TvSum50/RGB_as_numpy/' + new_fileName + '.npy', video)
        np.save('./../saved_numpy_arrays/' + new_fileName + '.npy', video)
        del video


#        np.save('../saved_numpy_arrays/targets_' + new_fileName + '.npy', targets)


if dataset == 'summe':
    processed_dataset = load_processed_dataset(PROCESSED_SUMME)
    mapped_video_names = read_json(SUMME_MAPPED_VIDEO_NAMES)
else:
    load_processed_dataset(PROCESSED_TVSUM)
    mapped_video_names = read_json(TVSUM_MAPPED_VIDEO_NAMES)
# RGB_numpy_arrays()


def arg_parser():
    # ../data/SumMe/videos  ../data/SumMe/GT
    # ../ data / TVSum / video /  ../data/TVSum/data
    # ../data/VSUMM/new_database  ../data/VSUMM/newUserSummary
    parser = argparse.ArgumentParser(description='Extract Features')
    parser.add_argument('--dataset', default='summe', type=str, help='summe, tvsum or vsumm')

RGB_numpy_arrays(fileNames)
