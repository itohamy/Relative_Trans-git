
import os,sys
sys.path.insert(1,os.path.join(sys.path[0],'..'))
import numpy as np
import matplotlib.pyplot as plt
from data_provider import DataProvider
from Edge import Edge
from FC_by_SIFT import get_FC_by_SIFT
from Plots import open_figure,PlotImages
import cv2
from image_warping import warp_image
from numpy.linalg import inv


def main():

    # ---------- Upload video frames: -----------------------------------------
    video_name = "movies/test_video.mp4"
    data = DataProvider(video_name)

    # ---------- Create set of edges E: ---------------------------------------
    print("Create set of edges E ...")
    E = create_E(data)
    print("Finished.")

    # ---------- Compute relative transformation for each edge in E: ----------
    print("Compute relative transformations for each edge in E ...")
    if os.path.exists('output_file'):
        os.remove('output_file')
    f = open('output_file','w+')
    for e in E:
        # Compute features correspondence by running SIFT:
        p, q, w = get_FC_by_SIFT(data.images[e.src], data.images[e.dst])

        # center the points around zero:
        img_sz = data.img_sz
        p[:,0] = p[:,0] - img_sz[1] / 2
        q[:,0] = q[:,0] - img_sz[1] / 2
        p[:,1] = p[:,1] - img_sz[0] / 2
        q[:,1] = q[:,1] - img_sz[0] / 2

        # Compute relative transformation (theta):
        E[e] = compute_LS_rigid_motion(p, q, w)
        # Add this measurement to the output file:
        pose = ' '.join([str(p) for p in np.reshape(E[e], (6,))])
        f.write('EDGE_SE2 ' + str(e.src) + ' ' + str(e.dst) + ' ' + pose + '\n')
    print("Finished.")

    #  ---------- view the relative transformation of a few edges:
    imgs = []
    titles = []
    for i,e in enumerate(E):
        if i in (200,201,203):
            imgs.append(data.images[e.src])
            imgs.append(data.images[e.dst])

            transformed_I = warp_image(data.images[e.dst], E[e], cv2.INTER_CUBIC)

            imgs.append(transformed_I)
            titles.append('I1')
            titles.append('I2')
            titles.append('theta(I2)')

    fig1 = open_figure(1, '', (5, 3))
    PlotImages(1, 3, 3, 1, imgs, titles, 'gray', axis=False, colorbar=False)
    plt.show()
    fig1.savefig('relative_trans_results.png', dpi=1000)


def create_E(data):
    E = dict()
    for i in range(data.feed_size-4):
        edge = Edge(i, i + 1)
        E[edge] = np.zeros((2,3))
        edge = Edge(i, i + 2)
        E[edge] = np.zeros((2,3))
        edge = Edge(i, i + 3)
        E[edge] = np.zeros((2,3))
        # edge = Edge(i,i + 4)
        # E[edge] = np.zeros((2,3))

    return E


# p_i.shape = 2x1; q_i.shape = 2x1; w_i is a scalar
# Returns theta (relative transformation) in shape 2x3
def compute_LS_rigid_motion(p, q, w):
    d = 2

    # Compute t & R:

    p_bar = np.sum(w * p, 0) / np.sum(w)
    q_bar = np.sum(w * q, 0) / np.sum(w)

    X = p - p_bar
    Y = q - q_bar
    W = np.diag(np.squeeze(w))

    S = X.transpose() @ W @ Y

    U, Sigma, V = np.linalg.svd(S)

    H = np.eye(d)
    H[d-1,d-1] = np.linalg.det(V @ U.transpose())

    R = V @ H @ U.transpose()
    t = q_bar - R @ p_bar

    upper_matrix = np.concatenate((R, np.expand_dims(t, axis=1)), axis=1)
    bottom_matrix = np.expand_dims(np.array([0,0,1]), axis=0)
    theta = np.concatenate((upper_matrix, bottom_matrix), axis=0)

    theta = inv(theta)

    return theta[0:2, :]



if __name__ == '__main__':
    main()


