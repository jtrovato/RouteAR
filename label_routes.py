# import the necessary packages
import argparse
import cv2
import pickle

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
routes = [[]]

def click_CB(event, x, y, flags, param):
    # grab references to the global variables
    global routes
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    #if event == cv2.EVENT_LBUTTONDOWN:

    # check to see if the left mouse button was released
    if event == cv2.EVENT_LBUTTONUP:
    # record the ending (x, y) coordinates and indicate that
    # the cropping operation is finished
        print("adding point")
        routes[-1].append((x, y))

    # draw the connecting line
        if len(routes[-1]) > 1:
            cv2.line(image, routes[-1][-2], routes[-1][-1], (0, 255, 0), 2)
            cv2.imshow("image", image)



####################   main script ####################
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
image = cv2.resize(image, (0,0), fx=0.25, fy=0.25)
clone = image.copy()
cv2.namedWindow("image")
cv2.resizeWindow('image', image.shape[0]/4, image.shape[1]/4)
cv2.setMouseCallback("image", click_CB)

# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)

    key = cv2.waitKey(1) & 0xFF

    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()

    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break

#write refernce points to a file
with open('routes.pickle', 'wb') as f:
    pickle.dump(routes, f)

# close all open windows
cv2.destroyAllWindows()
