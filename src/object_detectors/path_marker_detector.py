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

        @returns: Images associated to preprocessing and enhancement, segmentation, and bounding
        """
        pre  = super().preprocess(src)
        enh = super().enhance(pre, clahe_clr_spaces=[], clahe_clip_limit=3)
        seg = super().morphological(self.segment(enh), open_kernel=(3,3), close_kernel=(3,3))
        hulls = super().convex_hulls(seg, upper_area=1.0/2, lower_area=1.0/1000)
        bound = self.bound_path_marker(hulls, src)
        return enh,seg,bound


    def segment(self, src):
        """
        Segments the image using thresholded alpha channel

        @param src: A preprocessed image

        @returns: A segmented grayscale image
        """
        # Convert to LAB color space
        lab = cv.cvtColor(src, cv.COLOR_BGR2LAB)

        # Alpha threshold
        alpha_mean, alpha_std = cv.meanStdDev(lab[:,:,1])
        _,alpha_thresh = cv.threshold(lab[:,:,1], alpha_mean+2*alpha_std,255,cv.THRESH_BINARY)

        return alpha_thresh

    
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

        # We can't do anything if we aren't given any hulls
        if len(hulls) == 0:
            return src

        for hull in hulls:
            rect = cv.minAreaRect(hull)
            box = np.int0(cv.boxPoints(rect))
            src = cv.polylines(src, [box], True, (255,0,0), thickness=2)

        if self.debug:
            src = cv.polylines(src, hulls,True, (255,255,255),2)

        return src



