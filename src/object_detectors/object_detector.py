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
        Preprocesses the source image to adjust for underwater artifacts

        @param src: A raw unscaled image

        @returns: The preprocessed and scaled image
        """
        # Apply CLAHE and Gaussian on each RGB channel then resize
        clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
        bgr = cv.split(src)
        kernel = (3,3)
        bgr[0] = cv.GaussianBlur(clahe.apply(bgr[0]), kernel, 0) 
        bgr[1] = cv.GaussianBlur(clahe.apply(bgr[1]), kernel, 0) 
        bgr[2] = cv.GaussianBlur(clahe.apply(bgr[2]), kernel, 0) 
        src = cv.merge(bgr)
        self.im_dims = (int(src.shape[1]*self.im_resize), int(src.shape[0]*self.im_resize))
        src = cv.resize(src, self.im_dims, cv.INTER_CUBIC )
        self.curr_image = src
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