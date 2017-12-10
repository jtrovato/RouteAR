import sys, os
import cv2
import numpy as np
from matplotlib import pyplot as plt



#load images
print("loading images")
img1 = cv2.resize(cv2.imread('IMG_1541.JPG'), (0,0), fx=0.5, fy=0.5)
img2 = cv2.resize(cv2.imread('IMG_1542.JPG'), (0,0), fx=0.5, fy=0.5)
print(img1.shape)
gray1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)



#feature
print("finding features")
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1,None)
kp2, des2 = sift.detectAndCompute(gray2,None)

# BFMatcher with default params
"matching features"
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
plt.imshow(img3),plt.show()
