import cv2
import numpy as np
import utils

# finding the lane curves from the sampleTraining video
def getLaneCurve(img):
    imgThres = utils.thresholding(img)
    cv2.imshow('Thres', imgThres)
    return

if __name__ == "__main__":
    cap = cv2.VideoCapture('sampleTraining.mp4')
    while True:
        _, img = cap.read() # GET THE IMAGE
        img = cv2.resize(img,(640,480)) # RESIZE
        getLaneCurve(img)
        cv2.imshow('Vid', img)
        cv2.waitKey(1)
