#Created By: Logan Fillo
#Created On: 2020-03-25

import cv2 as cv
import numpy as np

from .object_detector import ObjectDetector


"""
Path Marker detector
"""


class PathMarkerDetector(ObjectDetector):
    """
    A class for detecting an underwater path marker, extends abstract object detector class
    """

    def __init__(self, im_resize=1.0, debug=False, focal=400):
        super().__init__(im_resize, debug, focal)
    

    def detect(self, src):
        pass

