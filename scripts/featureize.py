#Created By: Logan Fillo
#Created On: 2020-03-12

import numpy as np
import cv2 as cv


"""
Functions used to featureize convex hulls
"""


def featureize_model_data(data):
    """
    Featureizes the model data given in the form of a list of convex hull/label tuples
    and returns the feature array and label vector

    @param data: A list of tuples where each tuple is (convex hull, 1 if hull is pole, 0 otherwise)

    @returns: The feature matrix X, and label vector y
    """
    X = []
    y = []
    
    for d in data:
        hull, label = d
        X.append(form_feature_vector(hull))
        y.append(label)
    
    return np.asarray(X).astype(float), np.asarray(y).astype(float)


def featureize_hulls(hulls):
    """
    Featureizes the hulls and returns the features matrix X_hat

    @param hulls: The convex hulls to be featureized

    @returns: The feature matrix X_hat
    """
    X_hat = []  
    for hull in hulls:
        X_hat.append(form_feature_vector(hull))    
    return np.asarray(X_hat).astype(float)


def form_feature_vector(hull):
    """
    Forms the feature vector from a hull

    @param hull: The hull to featurize

    @returns: The feature vector of the hull
    """
    features = []

    MA, ma, angle = ellispe_features(hull)
    hull_area, rect_area, aspect_ratio = contour_features(hull)

    axis_ratio = float(MA)/ma
    angle = np.abs(np.sin(angle *np.pi/180))

    features.append(axis_ratio)
    features.append(aspect_ratio)
    features.append(angle)

    return np.asarray(features).astype(float)


def ellispe_features(hull):
    """
    Produces ellipse features of the hull

    @param hull: A convex hull 
    
    @returns: The major axis (MA), minor axis (ma) and angle of hull
    """
    angle = 0
    MA = 1
    ma = 1
    try:    
        # Fit ellipse only works if convex hull has 3 points
        (x,y),(MA,ma),angle = cv.fitEllipse(hull)
    except:
        pass
    return MA, ma, angle


def contour_features(hull):
    """
    Produces contour features of the hull

    @param hull: A convex hull
    
    @returns: The hull area, bounding rect area, aspect ratio
    """

    hull_area = cv.contourArea(hull)
    x,y,w,h = cv.boundingRect(hull)
    rect_area = w*h
    aspect_ratio =  float(w)/h
    return hull_area, rect_area, aspect_ratio