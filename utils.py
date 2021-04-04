import cv2
import numpy as np

# THRESHOLDING FUNCTION
def thresholding(img):
    # visualizing image in HSV parameters
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # the values for lowerWhite and upperWhite are found by tweaking the HSV min/max params in the 
    # trackbar by running ColorPickerScript.py
    lowerWhite = np.array([80, 0, 0])
    upperWhite = np.array([255, 160, 255])
    # passing the values of lowerWhite and upperWhite to create the mask
    maskWhite = cv2.inRange(imgHSV, lowerWhite, upperWhite)
    return maskWhite

# WARPING FUNCTION IMPLEMENTATION
# trackbar change ill call nothing()
def nothing(a):
    pass

def initializeTrackbars(intialTracbarVals,wT=480, hT=240):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0],wT//2, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], hT, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2],wT//2, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], hT, nothing)

def valTrackbars(wT=480, hT=240):
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([(widthTop, heightTop), (wT-widthTop, heightTop),
                      (widthBottom , heightBottom ), (wT-widthBottom, heightBottom)])
    return points