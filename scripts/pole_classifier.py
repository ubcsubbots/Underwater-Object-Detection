#Created By: Logan Fillo
#Created On: 2020-03-12

import pickle
import os
import numpy as np
import cv2 as cv
from sklearn import svm
import sklearn
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

from featureize import featureize_model_data


"""
Model training and plotting for pole convex hull classifier
"""


def plot(X,y, y_hat):
    """
    Assuming X is a (n,3) matrix, plots the examples of X in feature space where the points
    are BLUE if the example is correctly labelled as not a pole, RED if the example is 
    correctly labelled as a pole, and GREEN if the example isn't labelled correctly

    @param X: The feature vector (should be (n,3))
    @param y: The label vector (should be (n,))
    @param y_hat: The predicted labels of X

    """
    pole = X[(y == 1) & (y == y_hat)]
    not_pole = X[(y == 0) & (y == y_hat)]
    misclass = X[y != y_hat]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pole.T[0], pole.T[1], pole.T[2], c='r', label='Pole')
    ax.scatter(not_pole.T[0], not_pole.T[1], not_pole.T[2] ,label='Not Pole', c='b')
    ax.scatter(misclass.T[0], misclass.T[1], misclass.T[2], c='g', label='Misclassified')
    ax.legend()

    ax.set_xlabel("Axis Ratio")
    ax.set_ylabel("Aspect Ratio")
    ax.set_zlabel("Abs Sin of Angle")

    plt.show()


def train_and_plot():

    # Load pickle data
    directory = os.path.dirname(os.getcwd())
    with open(os.path.join(directory, 'pickle/pole_data2.pkl'), 'rb') as file:
        data = pickle.load(file)

        # Featurize raw data
        X, y = featureize_model_data(data)

        # We create test data set using an equal ratio of poles/not poles
        poles = X[y == 1]
        num_poles = poles.shape[0]
        not_poles = X[y == 0]
        num_not_poles = not_poles.shape[0]

        # Split data 4 ways
        poles_split = np.array_split(poles, 4)
        not_poles_split = np.array_split(not_poles, 4)

        # Train on 3/4 of the data
        X = np.vstack((poles_split[1], poles_split[2], poles_split[3], not_poles_split[1],not_poles_split[2], not_poles_split[3]))
        y = np.concatenate((np.ones((num_poles -poles_split[0].shape[0],)), np.zeros((num_not_poles -not_poles_split[0].shape[0], ))))

        # Test on 1/4 of the data
        X_test = np.vstack((poles_split[0], not_poles_split[0]))
        y_test = np.concatenate((np.ones((poles_split[0].shape[0],)), np.zeros((not_poles_split[0].shape[0], ))))

        # Train Model
        model = svm.SVC(kernel='linear')
        model.fit(X, y)

        # Get report on training data
        y_hat = model.predict(X)
        print("Training Report")
        print(classification_report(y, y_hat))

        # Get report on test data
        print("Test Report")
        y_hat_test = model.predict(X_test)
        print(classification_report(y_test, y_hat_test))

        # Pickle model
        d = os.path.dirname(os.getcwd())
        with open(os.path.join(d, 'pickle/model.pkl'), 'wb') as file:
            pickle.dump(model, file)
            print("Updated model.pkl")

        # Plot the classified training data
        plot(X,y, y_hat)

        # Plot the classified test data
        plot(X_test, y_test, y_hat_test )


if __name__ == '__main__':
    train_and_plot()