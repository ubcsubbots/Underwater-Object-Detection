#Created By: Logan Fillo
#Created On: 2020-03-25

import cv2 as cv
import numpy as np
from abc import ABC, abstractmethod


"""
Abstract base class for object detectors to implement
"""


class ObjectDetector(ABC):
    """
    Abstract base object detector class
    """


    def __init__(self, im_resize=1.0, debug=False, focal=400.0):
        """
        Initializes an object detector

        @param im_resize: The scale to resize the image to
        @param debug: If True, adds debug information to output
        """
        self.im_resize = im_resize
        self.im_dims = (0,0) # w, h
        self.debug = debug
        self.focal = focal # Camera focal length in pixels
        self.curr_image = None 


    @abstractmethod
    def detect(self, src):
        """
        Detects an object from a raw image 

        @param src: Raw image

        @returns: Three images representing the algorithm at various stages. The last image
                  must always be the final output of the algorithm
        """
        return src,src,src

    
    def preprocess(self, src):
        """
        Preprocesses the source image by blurring and resizing 

        @param src: A raw unscaled image

        @returns: The preprocessed and scaled image
        """
        self.im_dims = (int(src.shape[1]*self.im_resize), int(src.shape[0]*self.im_resize))
        if self.im_resize != 1.0:
            split = cv.split(src)
            kernel = (3,3)
            sig = 1
            split[0] = cv.GaussianBlur(split[0], kernel, sig)
            split[1] = cv.GaussianBlur(split[1], kernel, sig)
            split[2] = cv.GaussianBlur(split[2], kernel, sig)
            src = cv.merge(split)
            src = cv.resize(src, self.im_dims, cv.INTER_CUBIC )
            self.curr_image = src
        return src


    def enhance(self, src, clahe_clr_spaces=['bgr', 'hsv', 'lab'], clahe_clip_limit=1):
        """
        Enhances a raw image to account for underwater artifacts affecting contrast, hue and saturation. Performs
        CLAHE on the given input color spaces then blends the equally weighted result across all color spaces used

        @param src: A preprocessed image
        @param clahe_clr_spaces: The color spaces to perform CLAHE on
        @param clahe_clip_limit: The limit at which CLAHE clips the contrast to prevent over-contrasting
        
        @returns: An enhanced image
        """

        if any([s not in ['bgr', 'hsv', 'lab'] for s in clahe_clr_spaces]):
            print("Please only use any of ['bgr', 'hsv', 'lab'] as CLAHE color spaces")
            return src

        clahe = cv.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(11,11))
        parts = []

        # Apply CLAHE on all given CLAHE color spaces
        if 'bgr' in clahe_clr_spaces:
            bgr = cv.split(src)
            bgr[0] = clahe.apply(bgr[0])
            bgr[1] = clahe.apply(bgr[1])
            bgr[2] = clahe.apply(bgr[2])
            bgr_clahe = cv.merge(bgr)
            parts.append(bgr_clahe)
        if 'lab' in clahe_clr_spaces:
            lab = cv.cvtColor(src, cv.COLOR_BGR2LAB)
            lab = cv.split(lab)
            lab[0] = clahe.apply(lab[0])
            lab[1] = clahe.apply(lab[1])
            lab[2] = clahe.apply(lab[2])
            lab_clahe = cv.merge(lab)
            parts.append(cv.cvtColor(lab_clahe, cv.COLOR_LAB2BGR))
        if 'hsv' in clahe_clr_spaces: 
            hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
            hsv = cv.split(hsv)
            hsv[0] = clahe.apply(hsv[0])
            hsv[1] = clahe.apply(hsv[1])
            hsv[2] = clahe.apply(hsv[2])
            hsv_clahe = cv.merge(hsv)
            parts.append(cv.cvtColor(hsv_clahe, cv.COLOR_HSV2BGR))

        # Add parts using equal weighting
        if len(parts) > 0:
            weight = 1.0/len(parts)
            blended = np.zeros((self.im_dims[1], self.im_dims[0], 3))
            for p in parts:
                blended += weight*p
            src = blended.astype(np.uint8)
        return src


    def gradient(self, src):
        """
        Computes the sobel gradient of a source image

        @param src: A grayscale image

        @returns: The sobel gradient response of the image
        """
        scale = 1
        delta = 0
        ddepth = cv.CV_16S
        grad_x = cv.Sobel(src, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
        grad_y = cv.Sobel(src, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y)
        grad = np.expand_dims(cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0), axis=2)
        return grad


    def morphological(self, src, open_kernel=(1,1), close_kernel=(1,1)):
        """
        Smooths a binary image with morphological operations

        @param src: A binary image
        @param open_kernel: Opening kernel dimensions
        @param close_kernel: Closing kernel dimensions

        @returns: A morphologically smoothed image
        """
        # Opening followed by closing
        open_k = cv.getStructuringElement(cv.MORPH_RECT,open_kernel)
        close_k = cv.getStructuringElement(cv.MORPH_RECT,close_kernel)
        opening = cv.morphologyEx(src, cv.MORPH_OPEN, open_k)
        closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, close_k)
        return closing

    
    def convex_hulls(self, src, upper_area=1.0/2, lower_area=1.0/1000):
        """
        Creates a set of convex hulls from the binary segmented image and which are of an 
        appropriate size based on upper and lower area thresholds

        @params src: A binary segmented grayscale image
        @params upper_area: Upper threshold of area filter 
        @params lower_area: Lower threshold of area filter

        @returns: A set of convex hulls where each hull is an np array of 2D points
        """
        # Search over the binary images associated to each cluster
        hulls = []
        right_size_hulls = []

        # First find contours in the image
        _, contours, _ = cv.findContours(src, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Create a convex hull around each connected contour
        for j in range(len(contours)):
            hulls.append(cv.convexHull(contours[j], False))

        # Get the hulls whose area is within some thresholded range 
        for hull in hulls:
            hull_area = cv.contourArea(hull)
            im_size = self.im_dims[0]*self.im_dims[1]
            if (hull_area > im_size*lower_area and hull_area < im_size*upper_area):
                right_size_hulls.append(hull)

        return right_size_hulls