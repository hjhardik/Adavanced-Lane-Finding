import cv2
import numpy as np

# THRESHOLDING FUNCTION IMPLEMENTATION
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
def warpImg (img, points, w, h, inv=False):
    pts1 = np.float32(points)
    # defining the border coordinates of the warped image
    pts2 = np.float32([[0,0], [w,0], [0,h], [w,h]])
    # finding the transformation matrix
    if inv:
        #if inverted interchange pts2 and pts1
        matrix = cv2.getPerspectiveTransform(pts2,pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
    
    imgWarp = cv2.warpPerspective(img, matrix, (w,h))
    return imgWarp

# trackbar change will call nothing()
def nothing(a): 
    pass

# Creating the trackbars to find the optimal warping points.
# Care should be taken to choose points which are not very far from our current position
# ie. mostly lying in the bottom half region of the image since we should only confidently
# predict the lane warp present on the road at this point of time.

# create trackbars 
def initializeTrackbars(initialTrackbarVals, wT=480, hT=240): 
    # wT and hT are the target window dimensions ie. window with video
    # create trackbar window
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", initialTrackbarVals[0], wT//2, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", initialTrackbarVals[1], hT, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", initialTrackbarVals[2], wT//2, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", initialTrackbarVals[3], hT, nothing)

# find the value of trackbars (real-time)
def valTrackbars(wT=480, hT=240):
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    # return the bounding coordinates
    points = np.float32([(widthTop, heightTop), (wT-widthTop, heightTop), (widthBottom, heightBottom), (wT-widthBottom, heightBottom)])
    return points

# draw the warp points as red circles
def drawPoints(img, points):
    for x in range(0, 4):
        cv2.circle(img, (int(points[x][0]), int(points[x][1])), 12, (0,0,255), cv2.FILLED)
    return img

# HISTOGRAM IMPLEMENTATION (TO FIND CURVE TURNING LEFT/RIGHT)
def getHistogram(img, minPer=0.1, display= False, region=1): 
    # simply sum all the pixels in the y direction
    if region == 1:
        # find histvalues for the complete region
        histValues = np.sum(img, axis=0)
    else:
        # find histvalues for ONLY the bottom (1/n)th region where n is region value
        histValues = np.sum(img[img.shape[0]//region:,:], axis=0)
 
    #print(histValues)
    
    # Some of the pixels in our image might just be noise. So we don’t want to use them in our 
    # calculation. Therefore we will set a threshold value which will be the minimum value required
    # for any column to qualify as part of the path and not noise. We can set a hard-coded value but
    # it is better to get it based on the live data. So we will find the maximum sum value and 
    # multiply our user defined percentage to it to create our threshold value.
    maxValue = np.max(histValues)
    minValue = minPer*maxValue
    
    # To get the value of the curvature we will find the indices of all the columns that have value 
    # more than our threshold and then we will average our indices.
    indexArray = np.where(histValues >= minValue)
    basePoint = int(np.average(indexArray))
    #print(basePoint)
 
    if display:
        imgHist = np.zeros((img.shape[0],img.shape[1],3),np.uint8)
        for x,intensity in enumerate(histValues):
            cv2.line(imgHist,(x,img.shape[0]),(x,img.shape[0]-intensity//255//region),(255,0,255),1)
            cv2.circle(imgHist,(basePoint,img.shape[0]),20,(0,255,255),cv2.FILLED)
        return basePoint,imgHist
 
    return basePoint
 
# stack all the display windows
# (ONLY FOR DISPLAY PURPOSES, NO EFFECT ON PROGRAM) 
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver