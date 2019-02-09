
import os,sys
sys.path.insert(1,os.path.join(sys.path[0],'..'))
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from data_provider_Synthetic import DataProvider
from Edge import Edge
from Plots import open_figure,PlotImages
import cv2
from image_warping import warp_image


def main():

    # ---------- Upload video frames: -----------------------------------------
    crop_size_x = 600
    crop_size_y = 400
    data = DataProvider(crop_size_x, crop_size_y) # take just one photo and create synthetic movement

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
        # Add this measurement to the output file:
        pose = ' '.join([str(p) for p in np.reshape(E[e], (6,))])
        f.write('EDGE_SE2 ' + str(e.src) + ' ' + str(e.dst) + ' ' + pose + '\n')
    print("Finished.")

    #  ---------- view the relative transformation of a few edges:
    imgs = []
    titles = []
    for i,e in enumerate(E):
        if i in (120,121,122):
            imgs.append(data.imgs[e.src])
            imgs.append(data.imgs[e.dst])

            transformed_I = warp_image(data.imgs[e.dst], E[e], cv2.INTER_CUBIC)
            #transformed_I = cv2.warpAffine(data.imgs[e.src], E[e], (crop_size_x, crop_size_y), cv2.INTER_LANCZOS4)

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
    for i in range(data.feed_size-3):
        theta_rel1 = data.theta_rel[i]
        theta_rel1[0][2] = 2
        theta_rel2 = theta_rel1.copy()
        theta_rel2[0][2] = 4
        theta_rel3 = theta_rel1.copy()
        theta_rel3[0][2] = 6
        edge = Edge(i, i + 1)
        E[edge] = theta_rel1
        edge = Edge(i, i + 2)
        E[edge] = theta_rel2
        edge = Edge(i, i + 3)
        E[edge] = theta_rel3

    return E


if __name__ == '__main__':
    main()


