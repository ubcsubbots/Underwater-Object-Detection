#Created By: Logan Fillo
#Created On: 2020-03-12

import pickle
import os
import numpy as np
import cv2 as cv
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

from .featurize import PoleFeaturizer


"""
Model training and plotting for pole convex hull classifier
"""


class PoleClassifier:
    """
    A class for classifying pole convex hulls
    """


    def __init__(self, datafile="pole_data2.pkl"):
        self.featurizer = PoleFeaturizer()
        # Load pickle data
        directory = os.path.dirname(os.getcwd())
        with open(os.path.join(directory, 'pickle/' + datafile), 'rb') as file:
            self.data = pickle.load(file)


    def plot(self, X,y, y_hat):
        """
        Assuming X is a (n,3) or (n,2) matrix, plots the examples of X in feature space where the points
        are BLUE if the example is correctly labelled as not a pole, RED if the example is 
        correctly labelled as a pole, and GREEN if the example isn't labelled correctly

        @param X: The feature vector (should be (n,3))
        @param y: The label vector (should be (n,))
        @param y_hat: The predicted labels of X

        """
        _,d  = X.shape

        if (d != 2) and (d != 3):
            print("WARN: Must have 2 or 3 features to plot classification data")
            return

        pole = X[(y == 1) & (y == y_hat)]
        not_pole = X[(y == 0) & (y == y_hat)]
        misclass = X[y != y_hat]

        if d == 2:
            plt.scatter(pole.T[0], pole.T[1], c='r', label='Pole')
            plt.scatter(not_pole.T[0], not_pole.T[1], c='b', label='Not Pole')
            plt.scatter(misclass.T[0], misclass.T[1], c='g', label='Misclassified')
            plt.legend()

        elif d == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pole.T[0], pole.T[1], pole.T[2], c='r', label='Pole')
            ax.scatter(not_pole.T[0], not_pole.T[1], not_pole.T[2] ,label='Not Pole', c='b')
            ax.scatter(misclass.T[0], misclass.T[1], misclass.T[2], c='g', label='Misclassified')
            ax.legend()

        plt.show()


    def run(self):
        """
        Featurizes and trains an SVM classifier on pole convex hull data
        """

        # Featurize raw data
        X, y = self.featurizer.featurize_for_training(self.data)

        # Split the data 90/10 into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, train_size=0.9)

        # Train Model
        model = svm.SVC(kernel='rbf')
        model.fit(X_train, y_train)

        # Get report on training data
        y_hat = model.predict(X_train)
        print("Training Report")
        print(classification_report(y_train, y_hat))

        # Get report on test data
        print("Test Report")
        y_hat_test = model.predict(X_test)
        print(classification_report(y_test, y_hat_test))

        # Serialize model as pickle
        d = os.path.dirname(os.getcwd())
        with open(os.path.join(d, 'pickle/model.pkl'), 'wb') as file:
            pickle.dump(model, file)
            print("Updated model.pkl")

        # Plot the classified training data
        print("Plotting training data")
        self.plot(X_train,y_train, y_hat)

        # Plot the classified test data
        print("Plotting test data")
        self.plot(X_test, y_test, y_hat_test )
