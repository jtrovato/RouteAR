import sys, os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle
from shapely.geometry import Point,LineString

point_to_route_thresh = 10;
MIN_MATCH_COUNT = 10;

#load images and routes
print("loading images")
img1 = cv2.resize(cv2.imread('IMG_1541.JPG'), (0,0), fx=0.25, fy=0.25)
img2 = cv2.resize(cv2.imread('IMG_1546.JPG'), (0,0), fx=0.25, fy=0.25)
print(img1.shape)
gray1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

with open('routes.pickle', 'rb') as f:
    routes = pickle.load(f) #loads a list of LineStrings

#feature
print("finding features")
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1,None)
kp2, des2  = sift.detectAndCompute(gray2,None)

#lets get a list of features close to the route
# TODO: more effieicnet way of finding keypoint close to the route
# for route in routes:
#         route_kps = []
#         #get points of rect surronding line segement
#         for kp in kp1:
#             if Point(kp.pt).distance(route) < point_to_route_thresh:
#                 route_kps.append(kp)

print("number of kps: %d"%len(kp1))

#kp1, des1 = sift.compute(gray1, route_kps)
#kp2, des2 = sift.compute(gray2, kp2)

# BFMatcher with default params
"matching features"
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher()
matches = flann.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)
# cv2.drawMatchesKnn expects list of lists as matches.
# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
#img1 = cv2.drawKeypoints(gray1,route_kps,img1)


if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    route_pts = np.float32(np.array([list(pt) for pt in routes[0].coords])).reshape(-1,1,2)
    print(M)
    new_route = cv2.perspectiveTransform(route_pts, M)

    img2 = cv2.polylines(img2,[np.int32(new_route)],False,(0,255,0) ,3, cv2.LINE_AA)

else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None


plt.imshow(img2),plt.show()
