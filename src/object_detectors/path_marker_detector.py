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
        """
        Detects the path marker in a raw image and returns the images associated to the stages
        of the algorithm

        @param src: Raw underwater image containing the path marekr

        @returns: Images associated to preprocessing, segmentation, and bounding
        """
        pre  = super().preprocess(src)
        seg = super().morphological(self.segment(pre), open_kernel=(2,2), close_kernel=(2,2))
        hulls = super().convex_hulls(seg, upper_area=1.0/4, lower_area=1.0/800)
        bound = self.bound_path_marker(hulls, src)
        return pre,seg,bound


    def segment(self, src):
        """
        Segments the image using thresholded saturation gradient

        @param src: A preprocessed image

        @returns: A segmented grayscale image
        """
        # TODO: Implement better segmentaion strategy
        hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
        grad = super().gradient(hsv[:,:,1])
        _,grad_thresh = cv.threshold(grad, 0,255,cv.THRESH_OTSU)
        return grad_thresh

    
    def bound_path_marker(self, hulls, src):
        """
        Finds the convex hulls associated to the path marker and uses this to draw a bounding box around 
        the path marker in the raw image

        @param hulls: A set of the convex hulls to search
        @param src: The raw unscaled image 

        @returns: The raw scaled image with the bounding triangle around the path marker
        """
        
        # Resize src to match the image the hulls were found on
        src = cv.resize(src, self.im_dims, cv.INTER_CUBIC )

        # TODO: bound path marker

        return src



