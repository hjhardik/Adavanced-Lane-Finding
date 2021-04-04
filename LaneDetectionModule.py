import cv2
import numpy as np
import utils

# finding the lane curves from the sampleTraining video
def getLaneCurve(img):
    ##1 first find the threshold image
    imgThres = utils.thresholding(img)

    ##2 now find the warped tracking coordinates
    h, w, c = img.shape
    points = utils.valTrackbars()
    imgWarp = utils.warpImg(img, points, w, h)
    imgWarpPoints = utils.drawPoints(img, Points)
    
    # display the resultant images 
    cv2.imshow('Thres', imgThres)
    cv2.imshow('Warp', imgWarp)
    cv2.imshow('Warp Points', imgWarpPoints)
    return None

if __name__ == "__main__":
    # capture the training video of correct lane driving
    cap = cv2.VideoCapture('sampleTraining.mp4')
    frameCounter = 0
    while True:
        frameCounter +=1
        #keep repeating after video is over
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            frameCounter=0

        _, img = cap.read() # GET THE IMAGE
        img = cv2.resize(img,(640,480)) # RESIZE
        getLaneCurve(img)
        cv2.imshow('Vid', img)
        cv2.waitKey(1)
