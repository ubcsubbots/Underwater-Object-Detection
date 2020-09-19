#Created By: Logan Fillo
#Created On: 2020-03-12

import numpy as np
import cv2 as cv

from sklearn.preprocessing import StandardScaler


"""
Functions used to featurize object data for training and classification
"""


class PoleFeaturizer:
    """
    A class for constructing pole features for building models and classification
    """


    def __init__(self):
        self.scaler = StandardScaler()
        self.cnt_features = ContourFeatures()


    def featurize_for_training(self, data):
        """
        Featurizes the model data given in the form of a list of convex hull/label tuples
        and returns the feature array and label vector

        @param data: A list of tuples where each tuple is (convex hull, 1 if hull is pole, 0 otherwise)

        @returns: The feature matrix X, and label vector y used for training models
        """
        X = []
        y = []
        
        for d in data:
            hull, label = d
            X.append(self.form_feature_vector(hull))
            y.append(label)
        
        X = self.scale_data(np.asarray(X).astype(float))
        y = np.asarray(y).astype(float)
        return X, y


    def featurize_for_classification(self, hulls):
        """
        Featurizes the hulls and returns the features matrix X_hat

        @param hulls: The convex hulls to be featurized

        @returns: The feature matrix X_hat
        """
        X_hat = []  
        for hull in hulls:
            X_hat.append(self.form_feature_vector(hull))    

        X_hat = self.scale_data(np.asarray(X_hat).astype(float))
        return X_hat


    def form_feature_vector(self, hull):
        """
        Forms the feature vector from a convex hull

        @param hull: The convex hull to featurize

        @returns: The feature vector of the hull
        """
        features = []

        MA, ma, angle = self.cnt_features.ellispe_features(hull)
        hull_area, rect_area, aspect_ratio = self.cnt_features.area_features(hull)
        hu_moments = self.cnt_features.hu_moments_features(hull)
        min_rect, min_tri, min_circ_rad = self.cnt_features.min_area_features(hull)

        axis_ratio = float(MA)/ma
        angle = np.abs(np.sin(angle *np.pi/180))
        extent = cv.contourArea(min_rect)/hull_area
        triangularity = cv.contourArea(min_tri)/hull_area

        # Even though we calculate more features, from testing we find these work the best
        features.append(axis_ratio)
        features.append(aspect_ratio)
        features.append(extent)
        features += hu_moments

        x = np.asarray(features).astype(float)
        return x


    def scale_data(self, X, scale=False):
        """
        Scales data to normalize to 0 mean, 1 std
        
        @param X: unscaled feature matrix
        @param scale: If False, this method has no effect on X

        @returns: Scaled feature matrix 
        """
        if scale:
            self.scaler.fit(X)
            X = self.scaler.transform(X)
        return X


class ContourFeatures:
    """
    A class for contour features
    """
    
    
    def ellispe_features(self,cnt):
        """
        Produces ellipse features of the cnt

        @param cnt: A convex hull contour 
        
        @returns: The major axis (MA), minor axis (ma) and angle of contour
        """
        angle = 0
        MA = 1
        ma = 1
        try:    
            # Fit ellipse only works if convex hull has 3 points
            (x,y),(MA,ma),angle = cv.fitEllipse(cnt)
        except:
            pass
        return MA, ma, angle


    def area_features(self,cnt):
        """
        Produces area features of the contour

        @param cnt: A convex hull contour
        
        @returns: The contour area, bounding rect area, aspect ratio
        """

        cnt_area = cv.contourArea(cnt)
        x,y,w,h = cv.boundingRect(cnt)
        rect_area = w*h
        aspect_ratio =  float(w)/h
        return cnt_area, rect_area, aspect_ratio

    def min_area_features(self, cnt):
        """
        Produces min area features of the contour

        @param cnt: A convex hull contour

        @returns: Min area triangle, min area rect, min area circle radius
        """
        min_rect = cv.minAreaRect(cnt)
        min_rect = np.int0(cv.boxPoints(min_rect))
        min_tri = cv.minEnclosingTriangle(cnt)[1] # Sometimes returns None
        min_circ_rad = cv.minEnclosingCircle(cnt)[1]
        return min_rect, min_tri, min_circ_rad


    def hu_moments_features(self,cnt):
        """
        Produces the log of the hu moment features of the contour

        @param cnt: A convex hull contour

        @returns: A list of the 7 hu moments
        """
        # https://www.learnopencv.com/shape-matching-using-hu-moments-c-python/
        moments = cv.moments(cnt)
        mapped_hu_moments = map(lambda m: -1*np.copysign(1.0, m)*np.log10(abs(m)), cv.HuMoments(moments))
        hu_moments = np.array(list(mapped_hu_moments))

        return list(hu_moments.flatten())
