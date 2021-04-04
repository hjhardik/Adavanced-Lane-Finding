import cv2
import numpy as np
import utils

curveList = []
avgVal=10

# findings the lane curves from the sampleTraining video
def getLaneCurve(img, display=2):
    # create a copy image to display warp points
    imgCopy = img.copy()
    imgResult = img.copy()
    
    ##1 First find the threshold image
    imgThres = utils.thresholding(img)

    ##2 Now find the warped tracking coordinates
    hT, wT, c = img.shape
    points = utils.valTrackbars()
    imgWarp = utils.warpImg(imgThres, points, wT, hT)
    imgWarpPoints = utils.drawPoints(imgCopy, points)
    
    ##3 We have to find the center of the base which will give us the center line and 
    #   then compare the pixels on both side. By summation of these pixels we are basically
    #   finding the histogram
    middlePoint, imgHist = utils.getHistogram(imgWarp, display=True, minPer=0.5, region=4)
    # optimize curve for finding centre point for path incase of straight path but unsymmetrical histogram 
    curveAveragePoint, imgHist = utils.getHistogram(imgWarp, display=True, minPer=0.9)
    # subtract this value from the center to get the curve value. 
    curveRaw = curveAveragePoint - middlePoint

    ##4 Averaging the curve value for smooth transitioning
    curveList.append(curveRaw)
    if len(curveList)>avgVal:
        curveList.pop(0)
    curve = int(sum(curveList)/len(curveList))
 
    ##5 Display
    if display != 0:
        imgInvWarp = utils.warpImg(imgWarp, points, wT, hT, inv=True)
        imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
        imgInvWarp[0:hT // 3, 0:wT] = 0, 0, 0
        imgLaneColor = np.zeros_like(img)
        imgLaneColor[:] = 0, 255, 0
        imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
        imgResult = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)
        midY = 450
        cv2.putText(imgResult, str(curve), (wT // 2 - 80, 85), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
        cv2.line(imgResult, (wT // 2, midY), (wT // 2 + (curve * 3), midY), (255, 0, 255), 5)
        cv2.line(imgResult, ((wT // 2 + (curve * 3)), midY - 25), (wT // 2 + (curve * 3), midY + 25), (0, 255, 0), 5)
        for x in range(-30, 30):
            w = wT // 20
            cv2.line(imgResult, (w * x + int(curve // 50), midY - 10),
                     (w * x + int(curve // 50), midY + 10), (0, 0, 255), 2)
        #fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        #cv2.putText(imgResult, 'FPS ' + str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 50, 50), 3);
    if display == 2:
        # stack all the windows together (just for display purposes, no other requirement)
        imgStacked = utils.stackImages(0.7, ([img, imgWarpPoints, imgWarp],
                                             [imgHist, imgLaneColor, imgResult]))
        cv2.imshow('ImageStack', imgStacked)
    elif display == 1:
        cv2.imshow('Result', imgResult)
 
    ### NORMALIZATION
    curve = curve/100
    if curve>1: curve = 1
    if curve<-1: curve = -1
 
    return curve

if __name__ == "__main__":
    # capture the training video of correct lane driving
    cap = cv2.VideoCapture('sampleTraining.mp4')
    
    # initialize trackbar values for warping
    intialTrackbarVals = [102, 80, 20, 214]
    utils.initializeTrackbars(intialTrackbarVals)
    
    frameCounter = 0
    while True:
        frameCounter +=1
        # keep repeating after video is over
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            frameCounter=0

        _, img = cap.read() # GET THE IMAGE
        img = cv2.resize(img, (480,240)) # RESIZE
        # call getLaneCurve on the img
        curve = getLaneCurve(img, display=2)
        # print the stack of images
        print(curve)
        #cv2.imshow('Original Training Vid',img)
        cv2.waitKey(1)