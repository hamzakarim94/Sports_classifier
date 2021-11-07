import os
import tensorflow as tf

keras = tf.keras
from skimage import color
import cv2
import glob
import numpy as np


# import matplotlib.image as mpimg
def get_images(location, format, IMG_SIZE):
    image_array = []  # array which ll hold the images
    files = glob.glob("" + location + "*." + format + "")
    for myFile in files:
        image = cv2.imread(myFile)
        image_conv = tf.cast(image, tf.float32)
        image_conv = (image_conv / 127.5) - 1
        image_conv = tf.image.resize(image_conv, (IMG_SIZE, IMG_SIZE))
        # lum_img = image_conv[:, :, 0]
        image_conv = color.rgb2gray(image_conv)

        image_array.append(image_conv)  # append each image to array
    image_array = np.array(image_array)
    # image_lable.append(lable)
    print('image_array shape:', np.array(image_array).shape)
    # cv2.imshow('frame', image_array[0])
    # cv2.waitKey(0)
    return image_array



IMG_SIZE = 16
im_arr = get_images("MAZE2/", "png", IMG_SIZE)
c = 0
r = 1
def R():
    global c
    c=c + 1

def U():
    global r
    r =r - 1
def D():
    global r
    r= r + 1
def L():
    global c
    c=c - 1
policy = [R,U,D,L]#R,U,D,L #D,U,L,R
inv_policy = [L,D,U,R]#L,D,U,R #U,D,R,L
states = [[[] for _ in range(16)] for _ in range(16)]
checker = 0
im_arr[0, r, c] = 10
while c != 15 or r !=14 :

    r_prev =r
    c_prev =c
    for i in range(0,len(policy),1):
        #if checker == 0:
        policy[i]()
        if im_arr[0, r, c] > 0 and len(states[r][c]) == 0:
            im_arr[0, r, c] = 10
            inv_policy[i]()
            states[r][c].append(1)
            policy[i]()
        elif im_arr[0, r, c] < 0 and len(states[r][c]) == 0:
            im_arr[0, r, c] = -10
            inv_policy[i]()
            states[r][c].append(0)
        elif len(states[r][c]) == 4 and sum(states[r][c]) == 0:
            im_arr[0, r, c] = -10
            inv_policy[i]()
        elif len(states[r][c]) == 4 and sum(states[r][c]) > 0:
            for n in range(0 ,len(states[r][c]),1):
                if states[r][c] == 1:
                    policy[i]
                    break



