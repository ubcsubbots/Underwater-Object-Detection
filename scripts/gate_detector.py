#Created By: Logan Fillo
#Created On: 2020-03-12

import cv2 as cv
import numpy as np
np.set_printoptions(suppress=True)
from sklearn.preprocessing import normalize
from scipy.spatial.transform import Rotation as rot
import os
import pickle
import random

from featureize import featureize_hulls


"""
Gate detector using image segmentation, shape classification, and robust pose estimation
"""


class GateDetector:
    """
    A class for detecting an underwater gate
    """


    def __init__(self, im_resize=1.0, debug=False):
        self.im_resize = im_resize
        self.im_dims = (0,0) # w, h
        self.debug = debug
        self.gate_cntr = None
        self.gate_dims = (1.2192, 3.2004) # in m
        self.estimated_poses = []
        self.frame_count = 0
        self.gate_pose = (0.0,0.0,0.0,0.0,0.0,0.0) # x,y,z,phi,theta,psi
        self.focal = 400.0 # In pixels
        self.curr_image = None 

        directory = os.path.dirname(os.getcwd())
        with open(os.path.join(directory, 'pickle/model.pkl'), 'rb') as file:
            self.model = pickle.load(file)


    def detect(self, src):
        """
        Detects the gate in a raw image and returns the images associated to the stages
        of the algorithm

        @param src: Raw underwater image containing the gate

        @returns: Images associated to preprocessing, segmentation, bounding and pose estimation
        """
        pre = self.preprocess(src)
        seg = self.morphological(self.segment(pre))
        hulls = self.create_convex_hulls(seg)
        gate_im = self.bound_gate_using_poles(hulls, src)
        pose_im = self.estimate_gate_pose(gate_im)
        return pre, seg, pose_im

    def preprocess(self, src):
        """
        Preprocesses the source image to adjust for underwater artifacts

        @param src: A raw unscaled image

        @returns: The preprocessed image
        """
        # Apply CLAHE and Gaussian on each RGB channel then downsize
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

        @returns: The sobel gradient of the image
        """
        # Compute gradient using grayscale image
        scale = 1
        delta = 0
        ddepth = cv.CV_16S
        grad_x = cv.Sobel(src, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
        grad_y = cv.Sobel(src, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y)
        grad = np.expand_dims(cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0), axis=2)
        return grad


    def morphological(self, src):
        """
        Smooths a segmented image with morphological operations

        @param src: A segmented grayscale image

        @returns: A morphologically smoothed image
        """
        # Dilation then erosion to smooth segmentation
        dil_kernel = np.ones((1,1), np.uint8)
        ero_kernel = np.ones((1,1), np.uint8)
        dilated = cv.dilate(src, dil_kernel, iterations=1)
        eroded = cv.erode(dilated, ero_kernel, iterations=1)
        return eroded


    def segment(self, src):
        """
        Segments the image using thresholded saturation gradient and orange color mask

        @param src: A preprocessed image

        @returns: A segmented grayscale image
        """

        # Convert to HSV color space
        hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)

        # Compute gradient threshold on saturation channel of HSV image (seems to have best response to pole)
        grad = self.gradient(hsv[:,:,1])
        grad_mean, grad_std = cv.meanStdDev(grad)
        _,grad_thresh = cv.threshold(grad, grad_mean+4*grad_std,255,cv.THRESH_BINARY)
            
        # Create binary image of color mask on hue 
        upper_hue_mask = cv.inRange(hsv[:,:,0],175,180) # Upper orange/red range
        lower_hue_mask = cv.inRange(hsv[:,:,0],0,30) # Lower orange/red range
        color_mask = np.bitwise_or(upper_hue_mask, lower_hue_mask)

        # Combine the two binary image, note that the gradient threshold has the best response from farther away
        # and the color mask works best at close distances, so by combining them, we have an image that produces
        # a great response to the poles at all distances to them
        segmented = np.bitwise_or(grad_thresh, color_mask)

        return segmented


    def create_convex_hulls(self, src):
        """
        Creates a set of convex hulls from the binary segmented image and which are of an 
        appropriate size to be a pole of the gate

        @params src: A binary segmented grayscale image

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

        # Get the hulls whose area is within some reasonable range to be a pole
        for hull in hulls:
            hull_area = cv.contourArea(hull)
            im_size = self.im_dims[0]*self.im_dims[1]
            upper_range = 1.0/8
            lower_range = 1.0/800
            if (hull_area > im_size*lower_range and hull_area < im_size*upper_range):
                right_size_hulls.append(hull)

        return right_size_hulls


    def bound_gate_using_poles(self, hulls, src):
        """
        Finds the convex hulls associated to the poles and uses this to draw a bounding box around the poles 
        of the gate onto the raw image

        @param hulls: A set of the convex hulls to search
        @param src: The raw  unscaled image 

        @returns: The raw scaled image with the bounding box around the gate location drawn on
        """

        # Resize src to match the image the hulls were found on
        src = cv.resize(src, self.im_dims, cv.INTER_CUBIC )

        # We can't do anything if we aren't given any hulls
        if len(hulls) == 0:
            return src

        # Featureize hulls, predict using model and get classified pole hulls
        X_hat = featureize_hulls(hulls)
        y_hat = self.model.predict(X_hat)
        pole_hulls  = np.array(hulls)[y_hat == 1]

        # Get 2D array of all the points of the pole hulls (to determine extrema)
        hull_points = np.empty((0, 2))
        for hull in pole_hulls:
            hull = hull.reshape(hull.shape[0], 2)
            hull_points = np.vstack((hull_points,hull))

        # If we have detected a hull associated to a pole
        if len(hull_points > 0):

            gate_cntr, src = self.create_gate_contour(hull_points, src)

            # Get bounding box of contour to get it's approximate width/height
            _,_,w,h = cv.boundingRect(gate_cntr)

            # If the bounding rectangle is more wide than high, most likely we have detected both poles
            if float(w)/h >= 1:
                if (self.gate_cntr is not None):
                    # We make sure the area hasn't changed too much (50%) between frames to account for outliers 
                    prev_cntr = self.gate_cntr
                    prev_area = cv.contourArea(prev_cntr)
                    curr_area = cv.contourArea(gate_cntr)
                    if not (curr_area > 1.50*prev_area or curr_area < 0.50*prev_area):
                        self.gate_cntr = gate_cntr
                else:
                    # We make the strong assumption that the first time we detect the gate, it is accurate
                    self.gate_cntr = gate_cntr
                
        # Draw the gate if we have detected it
        if (self.gate_cntr is not None):
            src = cv.polylines(src, [self.gate_cntr], True, (0,0,255),2)

        # Draw all non pole hulls and pole hulls on src for debug purposes
        if self.debug:
            src = cv.polylines(src, hulls,True, (255,255,255),2)
            src = cv.polylines(src, pole_hulls,True, (0,0,255),2)

        return src


    def create_gate_contour(self, hull_points, src):
        """
        Creates the estimated gate contour from the given hull points, draws debug info on src if activated

        @param hull_points: 2D array of points
        @param src: The raw image 

        @returns: The gate contour drawn on the src image with debug info if activated 
        """

        width = self.im_dims[0]

         # Get extrema points of hulls (i.e the points closest/furthest from the top left (0,0) and top right (width, 0) of the image)
        top_left = hull_points[np.argmin(np.linalg.norm(hull_points, axis=1))]
        top_right = hull_points[np.argmin(np.linalg.norm(hull_points - np.array([width, 0]), axis=1))]
        bot_right = hull_points[np.argmax(np.linalg.norm(hull_points, axis=1))]
        bot_left = hull_points[np.argmax(np.linalg.norm(hull_points - np.array([width, 0]), axis=1))]

        if self.debug:
            # Draw extrema points for debug purposes
            src = cv.circle(src, (int(top_left[0]), int(top_left[1])), 8, (0,128,0), 4)
            src = cv.circle(src, (int(bot_right[0]), int(bot_right[1])), 8, (0,128,0), 4)
            src = cv.circle(src, (int(top_right[0]), int(top_right[1])), 8, (0,128,0), 4)
            src = cv.circle(src, (int(bot_left[0]), int(bot_left[1])), 8, (0,128,0), 4)

        gate_cntr = np.array([np.array([top_left], dtype=np.int32), 
                             np.array([top_right], dtype=np.int32), 
                             np.array([bot_right], dtype=np.int32), 
                             np.array([bot_left], dtype=np.int32)])
        return gate_cntr, src


    def estimate_gate_pose(self, src):
        """
        Estimates the gate pose by computing the median of calcualted poses across

        @param src: Image with gate contour drawn on

        @returns: Input image with pose estimation written on it
        """
        # Make sure we have a gate contour before estimating pose
        if self.gate_cntr is not None:
            
            # Estimate pose
            pose, src = self.calculate_gate_pose(src)
            self.estimated_poses.append(pose) 

            # Every 5 frames, set pose of gate as median of estimated poses, reset estimation frame
            if self.frame_count % 5 == 0:
                self.gate_pose = np.round(np.median(self.estimated_poses, axis=0), 2)
                self.estimated_poses = []

            # Draw gate pose
            x,y,z,phi,theta,psi = self.gate_pose
            text = "X:%.2fm, Y:%.2fm, Z:%.2fm, Roll:%.2fdeg, Pitch:%.2fdeg, Yaw:%.2fdeg" %(x,y,z,phi,theta,psi)
            w,h = self.im_dims
            text_point = (25, h-25)
            src = cv.putText(src, text, text_point,cv.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255))

        self.frame_count += 1
        return src


    def calculate_gate_pose(self, src):
        """
        Calculates the gate pose in 6 DOF using the gate contour

        @returns: A length 6 tuple containing the change in pose (x,y,z,phi,theta,psi) needed
                  to bring the AUV's body frame in line with the centre of the gate. Also returns
                  image with debug info on it if applicable

        """

        h,w = self.gate_dims
        top_left = self.gate_cntr[0][0]
        top_right = self.gate_cntr[1][0]
        bot_right = self.gate_cntr[2][0]
        bot_left = self.gate_cntr[3][0]

        # Calculate the horizontal and vertical components
        left_pole = bot_left - top_left
        right_pole = bot_right - top_right
        bot_line = bot_right - bot_left
        top_line = top_right - top_left

        # Use the top left point (closest to image origin) as basis origin
        origin = top_left
        hor  = top_line
        vert = left_pole

        A = np.power(hor[0],2) + np.power(hor[1],2)
        B = np.power(vert[0],2) + np.power(vert[1],2)
        C = hor[0]*vert[0] + hor[1]*vert[1]

        # Get the unique positive real root of the polynomial
        roots = np.roots([h**2, 0, (h**2*A-w**2*B), 0, -(w*C)**2])
        real_roots = roots[np.isreal(roots)].real
        pos_root =  real_roots[real_roots >=  0][0] # If root is zero, zero appears twice

        # Compute unrotated basis vectors
        u1 = np.concatenate((hor,[pos_root]))
        u2 = np.concatenate((vert, [-C/pos_root if pos_root > 0 else 0]))
        u3 = np.cross(u1,u2)

        # Get pixel scale information of 2D projection of 3D poles
        s = np.linalg.norm(u1)/w

        # Computed desired planar basis vectors
        v1 = [s*w,0,0]
        v2 = [0,s*h,0]
        v3 = [0,0,s*h*s*w]

        # Define normalized rotated basis (U) and desired unrotated basis (V)
        U = normalize(np.array([u1,u2,u3]).T, axis=0)
        V = normalize(np.array([v1,v2,v3]).T, axis=0)

        # Determine rotation matrix between bases and decompose into phi,theta,psi. Note that our order of basis
        # vectors is actually y,z,x so our decomposed euler angles are actually theta,psi,phi
        R = np.linalg.solve(U,V)
        q = rot.from_matrix(R)
        theta,psi,phi = tuple(q.as_euler('xyz', degrees=True))

        # The gate's width and height in terms of pixels
        w_prime = int(s*w)
        h_prime = int(s*h)

        # Get pixel width and height changes from image edge to a centered gate
        im_w, im_h = self.im_dims
        dw = (im_w - w_prime)//2
        dh = (im_h - h_prime)//2

        # Get desired pixel change in y,z to align centre of gate with AUV
        dy = dw - origin[0]
        dz = dh - origin[1]

        # Get x,y,z translational changes the AUV needs to take to bring it's body frame in
        # line with the center of the gate. Note that in the coordinate frame we are 
        # using, increasing x is forwards, increasing y is right, increasing z is down
        x = np.round(self.focal/s, 2)
        y = -np.round(dy*x/self.focal, 2)
        z = -np.round(dz*x/self.focal,2)

        if self.debug:
            # Gate's current basis
            src = cv.circle(src, (int(origin[0]), int(origin[1])), 8,  (0,128,0), 4)
            src = cv.line(src, tuple(bot_left), tuple(top_left),(0,128,0), 4)
            src = cv.line(src, tuple(top_right), tuple(top_left), (0,128,0), 2)
            # Gate's desired basis
            src = cv.circle(src, (int(dw), int(dh)), 8, (0,256,128), 4)
            src = cv.line(src, (dw,dh), (dw,dh+h_prime),(0,256,128), 4)
            src = cv.line(src, (dw,dh), (dw+w_prime, dh), (0,256,128), 2)
            # Line connecting translated origins
            src = cv.line(src, (origin[0],origin[1]), (origin[0]+dy, origin[1]+dz),(0,128,255), 4)

        return (x,y,z,phi,theta,psi), src




       