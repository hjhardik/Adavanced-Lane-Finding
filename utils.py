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
def warpImg (img,points,w,h,inv=False):

    pts1 = np.float32(points)
    # defining the border coordinates of the warped image
    pts2 = np.float32([[0,0], [w,0], [0,h], [w,h]])
    # finding the transformation matrix
    if inv:
        #if inverted interchange pts2 and pts1
        matrix = cv2.getPerspectiveTransform(pts2,pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
    
    imgWarp = cv2.warpPerspective(img,matrix,(w,h))
    return imgWarp

# trackbar change will call nothing()
def nothing(a): 
    pass

# Creating the trackbars to find the optimal warping points.
# Care should be taken to choose points which are not very far from our current position
# ie. mostly lying in the bottom half region of the image since we should only confidently
# predict the lane warp present on the road at this point of time.
def initializeTrackbars(intialTracbarVals,wT=480, hT=240): #wT and hT are the weight and height targets resp.
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