# import the necessary packages
import argparse
import cv2
import pickle
from shapely.geometry import Point, LineString

#TODO: label other thing other than routes! Use points on line string for bolts
# and anchors or ther things.
#TODO: multiple route labelling

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
routes = []
current_route = []

def click_CB(event, x, y, flags, param):
    # grab references to the global variables
    global routes
    global current_route
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    #if event == cv2.EVENT_LBUTTONDOWN:

    # check to see if the left mouse button was released
    if event == cv2.EVENT_LBUTTONUP:
    # record the ending (x, y) coordinates and indicate that
    # the cropping operation is finished
        print("adding point")
        current_route.append((x, y))

    # draw the connecting line
        if len(current_route) > 1:
            cv2.line(image, current_route[-2], current_route[-1], (0, 255, 0), 2)
            cv2.imshow("image", image)



####################   main script ####################
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
image = cv2.resize(image, (0,0), fx=1, fy=1)
clone = image.copy()
r = cv2.selectROI(clone)

image = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
cv2.namedWindow("image")
cv2.resizeWindow('image', image.shape[0]/1, image.shape[1]/1)
cv2.setMouseCallback("image", click_CB)

# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)

    key = cv2.waitKey(1) & 0xFF

    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()
    elif key == ord("b"):
        print("done bounding ROI")

    elif key == ord("n"):
        if current_route is not []:
            route = LineString(current_route)
            routes.append(route)
            current_route = []
    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        if current_route is not []:
            route = LineString(current_route)
            routes.append(route)
        break
        #TODO instead of closing, create another route.

#write refernce points to a file
with open('routes.pickle', 'wb') as f:
    pickle.dump(routes, f)
with open('crop.pickle', 'wb') as f:
    pickle.dump(r, f)

# close all open windows
cv2.destroyAllWindows()
