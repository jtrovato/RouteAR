import sys, os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle
from shapely.geometry import Point,LineString
import argparse

point_to_route_thresh = 10;
MIN_MATCH_COUNT = 10;

#parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_folder", required=True, help="Path to the images")
args = vars(ap.parse_args())

#load training image
print("loading training image")
print(os.path.join(args["image_folder"], 'im00000.jpg'))
train_im = cv2.imread(os.path.join(args["image_folder"], 'im00000.jpg'))
print(train_im.shape)
train_gray = cv2.cvtColor(train_im,cv2.COLOR_BGR2GRAY)

#crop the training image
with open('crop.pickle', 'rb') as f:
    r = pickle.load(f) #loads a perspectiveTransform
train_gray = train_gray[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

sift = cv2.xfeatures2d.SIFT_create()
kp_train, des_train = sift.detectAndCompute(train_gray,None)

#load routes
with open('routes.pickle', 'rb') as f:
    routes = pickle.load(f) #loads a list of LineStrings

im_ending = ["peg", "jpg", "png", "PNG", "PEG", "JPG"]
im_filenames = [os.path.join(args["image_folder"], f) for f in os.listdir(args["image_folder"]) if f[-3:] in im_ending]
print(im_filenames)

for im in im_filenames:
    #load image
    print(im)
    img = cv2.imread(im)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #feature
    print("finding features")
    kp, des  = sift.detectAndCompute(gray,None)

    #lets get a list of features close to the route
    # TODO: more effieicnet way of finding keypoint close to the route
    # for route in routes:
    #         route_kps = []
    #         #get points of rect surronding line segement
    #         for kp in kp1:
    #             if Point(kp.pt).distance(route) < point_to_route_thresh:
    #                 route_kps.append(kp)

    print("number of kps: %d"%len(kp))

    #kp1, des1 = sift.compute(gray1, route_kps)
    #kp2, des2 = sift.compute(gray2, kp2)

    # BFMatcher with default params
    print("matching features")
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(des_train, des, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    # cv2.drawMatchesKnn expects list of lists as matches.
    # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
    #img1 = cv2.drawKeypoints(gray1,route_kps,img1)


    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp_train[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        for route in routes:
            route_pts = np.float32(np.array([list(pt) for pt in route.coords])).reshape(-1,1,2)
            new_route = cv2.perspectiveTransform(route_pts, M)
            img = cv2.polylines(img,[np.int32(new_route)],False,(0,255,0) ,3, cv2.LINE_AA)

    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    #draw lines on original
    for route in routes:
        train_im = cv2.polylines(train_im, [np.int32(route)], False, (0,255,0), 3, cv2.LINE_AA)



    plt.figure()
    plt.imshow(img)
plt.figure("train")
plt.imshow(train_im)
plt.show()
