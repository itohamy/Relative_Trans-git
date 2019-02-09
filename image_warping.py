
import numpy as np
from numpy.linalg import inv
import cv2


# T: 2x3 affine matrix, which is the INVERSE map!
# Assume that the rotation in T is around the middle of the image.
def warp_image(I, T, interpolation = cv2.INTER_CUBIC):
    height, width, _ = I.shape

    # convert the transformation to be rotated around the image center
    T = get_centered_transform_for_remap(T, height, width)

    T = invert_T(T)

    map_x, map_y = create_mappings(T, height, width)
    I_warpped_remap = cv2.remap(I, map_x, map_y, interpolation)
    return I_warpped_remap


def create_mappings(T, height, width):
    yy, xx = np.mgrid[:height, :width]
    yy_pts = np.reshape(yy, (1, height*width))
    xx_pts = np.reshape(xx, (1, height*width))
    ones_line = np.ones((1, height*width))
    xy_pts = np.concatenate((xx_pts, yy_pts, ones_line), axis=0)
    xy_pts_trans = T @ xy_pts
    map_x = np.reshape(xy_pts_trans[0, :], (height, width))
    map_y = np.reshape(xy_pts_trans[1, :], (height, width))
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    return map_x, map_y


# Input: transformation T, 2x3
# Output: transformation T^{-1}, 2x3
def invert_T(T):
    ones_line = np.array([[0,0,1]])
    T = np.concatenate((T, ones_line),axis=0)
    T = inv(T)
    return T[0:2,:]


def get_centered_transform_for_remap(T, height, width):
    tx_gal = -((width//2)*T[0][0]) + ((height//2)*T[1][0]) + (width//2) + T[0][2]
    ty_gal = -((width//2)*T[1][0]) - ((height//2)*T[0][0]) + (height//2) + T[1][2]
    T[0][2] = tx_gal
    T[1][2] = ty_gal
    return T