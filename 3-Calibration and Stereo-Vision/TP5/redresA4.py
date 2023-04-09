#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 13:49:38 2022

@author:
"""


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow, show, subplot, title, axis, figure, plot


# draw matches
def draw_matches(img2, kpt2, matches,rayon):

    h2, w2 = img2.shape[:2]

    # Create a blank image with the size of the first image + second image
    new_img = np.zeros((h2, w2, 3), dtype='uint8')
#    new_img[:h2, :w2, :] = np.dstack([img2, img2, img2])
    new_img[:h2, :w2, :] = np.dstack([img2])

    # extract the match keypoints
    
    for m in matches:
        (x2, y2) = kpt2[m.trainIdx].pt

        # Draw circles on the keypoints
        cv.circle(new_img, (int(x2), int(y2)), rayon, (255, 255, 255), 1)

    return new_img


def warpImages(kp, img, M):
    # get the corner coordinates of the "query" and "train" image
    h=kp[2].pt[1]
    w=kp[2].pt[0]
            
    new_img = cv.warpPerspective(img, M,(np.int32(w),np.int32(h)))

    return new_img


def find_homography(kpt1, kpt2, matches):
    # Find an Homography matrix between two pictures
    # Transforming keypoints to list of points
   
    dst_pts = np.float32([kpt1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    src_pts = np.float32([kpt2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
 
    # Compute a rigid transformation (only scale + rotation + translation) /affine transformation
    affine_matrix, rigid_mask = cv.estimateAffinePartial2D(src_pts, dst_pts)    
    affine_row = [0, 0, 1]
    affine_matrix = np.vstack((affine_matrix, affine_row))
    
    M=affine_matrix
    print(M)

    # Compute an homography
        
    M, mask = cv.findHomography(src_pts, dst_pts,cv.RANSAC,3)

    print(M)
    print(mask)
    
    return M
    #return affine_matrix


# Read images

img1c = cv.imread('f7.jpg', 1)


img1 = cv.cvtColor(img1c, cv.COLOR_BGR2GRAY)

img1c = cv.cvtColor(img1c, cv.COLOR_BGR2RGB)


Noutliers=1

# Données pour l'image redressée

resol=5
x1=0
y1=0
sizeH=210
sizeV=297
x2=resol*sizeH
y2=resol*sizeV

pts1=[]
pts1.append([x1,y1])
pts1.append([x2,y1])
pts1.append([x2,y2])
pts1.append([x1,y2])




demil=(x1+x2)/4
demih=(y1+y2)/4

for i in range(Noutliers):
    pts1.append([demil+demil*np.random.randn(),demih+demih*np.random.randn()])


print('pts1',pts1)




# données pour f6.jpg

x1=248
y1=949
x2=1450
y2=924
x3=1541
y3=2592
x4=225
y4=2621


# données pour f7.jpg

x1=2030
y1=110
x2=2750
y2=904
x3=920
y3=1638
x4=363
y4=663


demil=(x1+x2+x3+x4)/8
demih=(y1+y2+y3+y4)/8

pts2=[]
pts2.append([x1,y1])
pts2.append([x2,y2])
pts2.append([x3,y3])
pts2.append([x4,y4])

for i in range(Noutliers):
    pts2.append([demil+demil*np.random.randn(),demih+demih*np.random.randn()])

print('pts2', pts2)


kp1 = [cv.KeyPoint(x[0], x[1], 1) for x in pts1]
kp2 = [cv.KeyPoint(x[0], x[1], 1) for x in pts2]

print('points feuille')
for i in range (len(pts1)):
    print(kp1[i].pt)

print('points image')
for i in range (len(pts2)):
    print(kp2[i].pt)


goodMatches=[]
for i in range(len(pts1)):
    m=cv.DMatch(i,i,0)
    goodMatches.append(m)

print('matchess')
for m in goodMatches:
    (x1, y1) = kp1[m.queryIdx].pt
    (x2, y2) = kp2[m.trainIdx].pt
    print ( x1,y1,x2,y2)
print('points')

figure()

kp_img = cv.drawKeypoints(img1, kp2, None, color=(0, 255, 0))


imshow(kp_img)
show()


#
## find  affine transformation and panoramic view between 1 to 2
matrix1to2 = find_homography(kp1, kp2,goodMatches)
img1to2 = warpImages(kp1,img1c, matrix1to2)
#

figure()

imshow(img1to2)
show()












