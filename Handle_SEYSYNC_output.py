
import os,sys
sys.path.insert(1,os.path.join(sys.path[0],'..'))
import numpy as np
import matplotlib.pyplot as plt
from data_provider import DataProvider
from Plots import open_figure,PlotImages
from numpy.linalg import inv
import cv2
from image_warping import warp_image


def main():

    # ---------- Upload video frames: -----------------------------------------
    video_name = "movies/test_video.mp4"
    data = DataProvider(video_name)

    # ---------- Read the SE-Sync output: ---------------------------------------
    V = create_V(data)
    f = open("SE_Sync_output.txt", "r")
    for line in f:
        tokens = line[:-1].split(' ')
        tokens = [float(i) for i in tokens]
        v = int(tokens[0])
        V[v] = extract_optimal_pose(tokens[1:])

    #  ---------- Transform the images and create a panorama: --------------------
    big_I_sz = ((1000, 2000))
    imgs_orig = []
    imgs_trans = []
    imgs_trans_all = []
    titles = []
    panoramas = []

    T1_inv = invert_T(V[0])
    for v in V:
        if v in range(0,9):
            img = data.images[v]
            # print(V[v])
            T = revert_T(V[v], T1_inv)

            img_sz = img.shape
            # embed the image:
            I = np.zeros((big_I_sz[0],big_I_sz[1],3))
            I[big_I_sz[0] // 4:big_I_sz[0] // 4 + img_sz[0],big_I_sz[1] // 4:big_I_sz[1] // 4 + img_sz[1],:] = img
            I = I / 255.

            if np.isfinite(np.linalg.cond(T)):
                I_Rt = warp_image(I, T, cv2.INTER_CUBIC)
                I_Rt = np.abs(I_Rt)
                imgs_trans_all.append(I_Rt)
                # view the global transformation of a few vertices:
                if v in range(0, 9):
                    print(T)
                    imgs_orig.append(data.images[v])
                    imgs_trans.append(I_Rt)
                    titles.append('')

    # build panoramic image:
    y = nan_if(imgs_trans_all)
    panoramic_img = np.nanmean(y, axis=0)
    panoramas.append(panoramic_img)

    fig1 = open_figure(1,'',(5,3))
    PlotImages(1, 3, 3, 1,imgs_orig,titles,'gray',axis=False,colorbar=False)
    fig2 = open_figure(2,'',(5,3))
    PlotImages(2, 3, 3, 1,imgs_trans,titles,'gray',axis=False,colorbar=False)
    fig3 = open_figure(3,'Panoramic Image', (5,5))
    PlotImages(3,1,1,1,panoramas,titles,'gray',axis=True,colorbar=False)
    fig1.savefig('optimal_trans_results_orig.png', dpi=1000)
    fig2.savefig('optimal_trans_results.png', dpi=1000)
    fig3.savefig('Panorama.png', dpi=1000)
    plt.show()


def create_V(data):
    V = dict()
    for i in range(data.feed_size):
        V[i] = np.zeros((2,3))
    return V


def extract_optimal_pose(tokens):
    poses = np.asarray(tokens)
    return np.reshape(poses, (2, 3))


# Input: transformation T, 2x3
# Output: transformation T^{-1}, 2x3
def invert_T(T):
    ones_line = np.array([[0,0,1]])
    T = np.concatenate((T, ones_line), axis=0)
    T = inv(T)
    return T[0:2, :]


# Input: 2 transformations T and T1, 2x3.
# Output: transformation ~T, 2x3
def revert_T(T, T1_inv):
    ones_line = np.array([[0,0,1]])
    T1_inv = np.concatenate((T1_inv, ones_line), axis=0)
    T = np.concatenate((T, ones_line), axis=0)
    T_gal = T1_inv @ T
    return T_gal[0:2, :]


# Input: list of elements
# Output: list of elements after replacing small values with nan
def nan_if(lst):
    new_lst = []
    for i in range(len(lst)):
        I = lst[i]
        I_new = np.where(I <= 1e-04, np.nan, I)
        #I_new = np.where(I_new == 1, np.nan, I_new)
        new_lst.append(I_new)
    return new_lst


if __name__ == '__main__':
    main()



# OLD:
# R = cv2.getRotationMatrix2D((2000 / 2,1000 / 2), np.rad2deg(np.arccos(p[0])),1)
# I_r = cv2.warpAffine(I,R,(2000,1000))
# t = np.float32([[1,0,p[2]],[0,1,p[5]]])
# I_Rt = cv2.warpAffine(I_r,t,(2000,1000))
#transformed_I = cv2.warpAffine(I, V[v], (2000, 1000), cv2.INTER_LANCZOS4)