**Lane Detection Algorithm for Autonomous Vehicles using Computer Vision (OpenCV)**

**Directions for use**:

_MainRobot.py, MotorModule.py, WebcamModule.py scripts will only needed when connecting to Raspberry Pi using RPi._

1. Fork and Clone the repo and navigate to the directory using cmd.
2. Run LaneDetectionModule.py using python <_filename_> command. (Check if your installed python version is compatible with OpenCV 2.0)
3. Tune the parameters accordingly in order to mark the best ROI.

_for visualizing best HSV colorpicking in the thresholding part:
python ColorPickerScript.py
(in the main directory)_

**Final output will appear like this:**

![final result gif](https://github.com/hjhardik/Advanced-Lane-Finding/blob/master/gifs/allFinal.gif)
