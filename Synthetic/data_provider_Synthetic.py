import numpy as np
import os
import shutil
import glob
import cv2
import random
import matplotlib.pyplot as plt
import math
from scipy.linalg import logm
import matplotlib.pyplot as plt


class DataProvider:

    def __init__(self, crop_size_x, crop_size_y):

        print('Start loading data ...')

        # prepare cropped images from all data set
        self.imgs = []
        self.theta_rel = []
        self.theta_glob = []

        img_str = "Data/frame1.jpg"
        I = cv2.imread(img_str)
        I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        img_sz = I.shape
        print("image size: ", img_sz)

        t1 = 0
        t2 = 0
        while (t1+crop_size_x <= img_sz[1] and t2+crop_size_y <= img_sz[0]):

            # Synthetic movement (t1,t2,r)
            t1 = t1 + 2
            t2 = 0
            rot = 0

            # if there is a rotation:
            # M = cv2.getRotationMatrix2D((img_sz[1] / 2, img_sz[0] / 2), np.rad2deg(rot), 1)
            # I = cv2.warpAffine(I,M,(img_sz[0],img_sz[1]), cv2.INTER_LANCZOS4)

            img = I[t2:t2+crop_size_y, t1:t1+crop_size_x, :]

            # ----- create global ground-truth theta:
            theta_glob = cv2.getRotationMatrix2D((img_sz[1] / 2,img_sz[0] / 2),np.rad2deg(rot),1)
            theta_glob[0][2] = float(t1)
            theta_glob[1][2] = float(t2)
            # ----- create relative ground-truth theta (between I and I+1):
            theta_rel = theta_glob
            theta_rel[0][2] = 2

            self.imgs.append(img)
            self.theta_rel.append(theta_rel)
            self.theta_glob.append(theta_glob)

            self.feed_size = len(self.imgs)

        print('Finished uploading data, number of frames:', len(self.imgs))


def makedir(folder_name):
    try:
        if os.path.exists(folder_name) and os.path.isdir(folder_name):
            shutil.rmtree(folder_name)
        os.makedirs(folder_name)
    except OSError:
        pass
    # cd into the specified directory
    # os.chdir(folder_name)