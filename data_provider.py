import numpy as np
import os
import shutil
import glob
import cv2
from video_to_frames import extractImages
import matplotlib.pyplot as plt


class DataProvider:

    def __init__(self, video_name):

        print('Start loading data ...')
        self.feed_path = "Data"

        # load data from video
        makedir(self.feed_path)
        self.feed_size = extractImages(video_name, self.feed_path)

        # prepare cropped images from all data set
        files = glob.glob(self.feed_path + "/*.jpg")
        self.images = []
        for img_str in files:
            I = cv2.imread(img_str)
            I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
            #I = cv2.resize(I, (img_sz, img_sz))
            #I = I / 255.
            self.images.append(I)

        self.img_sz = I.shape
        # plt.imshow(self.images[0])
        # plt.show()
        print('Finished uploading data, Number of video frames:', self.feed_size)

    # def next_batch(self, batch_size, data_type):
    #     batch_x = None
    #     batch_theta_gt = None
    #     if data_type == 'train':
    #         idx = np.random.choice(self.train_size, batch_size) # np.arange(25)
    #         batch_x = self.train[idx, ...]
    #         batch_x_crop = self.train_crop[idx, ...]
    #         batch_theta_gt = self.train_theta[idx, ...]
    #     elif data_type == 'test':
    #         idx = np.random.choice(self.test_size, batch_size)  # np.arange(25)
    #         batch_x = self.test[idx, ...]
    #         batch_x_crop = self.test_crop[idx,...]
    #         batch_theta_gt = self.test_theta[idx,...]
    #     return batch_x, batch_x_crop, batch_theta_gt


def makedir(folder_name):
    try:
        if os.path.exists(folder_name) and os.path.isdir(folder_name):
            shutil.rmtree(folder_name)
        os.makedirs(folder_name)
    except OSError:
        pass
