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
options = [[[] for _ in range(16)] for _ in range(16)]
checker = 0


#FORWARD SOLVE
###############################################
im_arr[0, r, c] = 10
states[r][c].append('V')
countf = 0
while c != 15 or r !=14 :
    countf = countf+1
    r_prev =r
    c_prev =c
    ''' for i in range(0,len(policy),1):
        #if checker == 0:
        policy[i]()
        if im_arr[0,r,c] > 0 and r>=0 and c>=0:
            inv_policy[i]()
            options[r][c].append(1)
        elif im_arr[0,r,c] < 0 and r>=0 and c>=0:
            inv_policy[i]()
            options[r][c].append(0)

    r = r_prev
    c = c_prev'''
    for i in range(0,len(policy),1):
        #if checker == 0:
        policy[i]()

        #else:
            #inv_policy[i]()
        if im_arr[0,r,c] > 0 and len(states[r][c]) == 0:
            states[r][c].append('V')
            #inv_policy[i]()#go back
            states[r][c].append(i)
            #policy[i]()
            im_arr[0, r, c] = 10

            break

        elif im_arr[0, r, c] > 0 and states[r][c][0] == 'V' and i <3:
            if i not in states[r][c]:
                states[r][c].append(i)
            inv_policy[i]()
        elif im_arr[0, r, c] > 0 and states[r][c][0] == 'V' and i == 3:
            inv_policy[i]()
            inv_policy[i]()
            if im_arr[0, r, c] < 0:
                policy[i]()
                p = states[r][c]
                states[r][c][0] = 'B'
                im_arr[0, r, c] = -10

            else:
                policy[i]()
            policy[i]()

        elif im_arr[0, r, c] < 0 and i == 3:
            print(r,c)
            inv_policy[i]()
            print(r, c)
            states[r][c][0] = 'B'
            im_arr[0, r, c] = -10
            p = states[r][c][len(states[r][c])-1]
            inv_policy[states[r][c][len(states[r][c])-1]]()
            states[r][c].pop()
            print()
        else:
            if 'B' not in states[r][c]:
                states[r][c].append('B')
            im_arr[0, r, c] = -10
        #if checker == 0:
            inv_policy[i]()



#BACK SOLVE
############################################
r=14
c=15
options[r][c].append(0)
options[r][c].append(0)
options[r][c].append(0)
options[r][c].append(1)
states2 = [[[] for _ in range(16)] for _ in range(16)]
states2[r][c].append('V')
countb = 0
while c != 0 or r !=1 :
    countb = countb + 1
    r_prev =r
    c_prev =c
    for i in range(0,len(policy),1):
        #if checker == 0:
        inv_policy[i]()#go forward
        if r>15 or c>15 or r<0 and c<0:
            policy[i]()#go back
        #else:
            #inv_policy[i]()
        if im_arr[0,r,c] > 0 and len(states2[r][c]) == 0:
            states2[r][c].append('V')
            states2[r][c].append(i)
            im_arr[0, r, c] = 10

            break

        elif im_arr[0, r, c] > 0 and states2[r][c][0] == 'V' and i <3:
            if i not in states2[r][c]:
                states2[r][c].append(i)
            policy[i]()#go back
        elif im_arr[0, r, c] > 0 and states2[r][c][0] == 'V' and i == 3:
            policy[i]()
            policy[i]()
            if im_arr[0, r, c] < 0:
                inv_policy[i]()
                p = states2[r][c]
                states2[r][c][0] = 'B'
                im_arr[0, r, c] = -10

            else:
                inv_policy[i]()
            inv_policy[i]()

        elif im_arr[0, r, c] < 0 and i == 3:
            print(r,c)
            policy[i]()
            print(r, c)
            states2[r][c][0] = 'B'
            im_arr[0, r, c] = -10
            p = states2[r][c][len(states2[r][c])-1]
            policy[states2[r][c][len(states2[r][c])-1]]()
            states2[r][c].pop()
            print()
        else:
            if 'B' not in states2[r][c]:
                states2[r][c].append('B')
            im_arr[0, r, c] = -10
        #if checker == 0:
            policy[i]()#go back


#STATES AND MDP SETUP
#############################################################
MDP = [[[] for _ in range(16)] for _ in range(16)]
MDP[14][15].append(100)
for R in range (0,15,1):
    r =R
    for C in range(0, 15, 1):
        r =R
        c =C
        if im_arr[0, r, c] == 10:
            for i in range(0, len(policy), 1):
                c_prev=c
                r_prev =r
                # if checker == 0:
                policy[i]()

                if  r >= 0 and c >= 0 and r <= 15 and c <= 15 and im_arr[0, r, c] == 10 and len(options[r_prev][c_prev]) < 4:
                    inv_policy[i]()
                    if len(MDP[r][c]) == 0:
                        MDP[r][c].append(0)

                    options[r][c].append(1)
                elif  r >= 0 and c >= 0 and r <= 15 and c <= 15 and im_arr[0, r, c] < 10 and len(options[r_prev][c_prev]) < 4:
                    inv_policy[i]()
                    options[r][c].append(0)
                elif r <= 0 or c <= 0 or r >= 15 or c >= 15:
                    inv_policy[i]()
                    options[r][c].append(0)


#MDP IMPLEMENTATION
##########################################
for s in range(0,16*16,1):
    g = 0.5
    e =1
    for R in range(0, 15, 1):
        r = R
        for C in range(0, 15, 1):
            r = R
            c = C
            if im_arr[0, r, c] == 10:
                for i in range(0, len(policy), 1):
                    c_prev = c
                    r_prev = r
                    # if checker == 0:
                    policy[i]()

                    if r >= 0 and c >= 0 and r <= 15 and c <= 15 and len(MDP[r][c])> 0 and im_arr[0,r,c] == 10:
                        p = MDP[r][c][0]
                        inv_policy[i]()
                        MDP[r][c][0] =g*(MDP[r][c][0] + e*p)

                    else:
                        inv_policy[i]()

#SHORTEST PATH USING MDP
########################################################
p =0
c = 0
r =1
im_arr[0,r,c] = 100
while c != 15 or r !=14 :
    for i in range(0, len(policy), 1):
        c_prev = c
        r_prev = r
        # if checker == 0:
        if options[r][c][i] == 1:
            policy[i]()
            if r >= 0 and c >= 0 and r <= 15 and c <= 15 and len(MDP[r][c])> 0:
                if MDP[r][c][0] > p:
                    Rp = r
                    Cp =c
                    p = MDP[r][c][0]
            inv_policy[i]()
    im_arr[0,Rp,Cp] = 100
    r=Rp
    c=Cp
print(im_arr)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
