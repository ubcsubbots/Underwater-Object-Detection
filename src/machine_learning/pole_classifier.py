#Created By: Logan Fillo
#Created On: 2020-03-12

import pickle
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
 

from .featurize import PoleFeaturizer


"""
Model training and plotting for pole convex hull classifier
"""


class PoleClassifier:
    """
    A class for classifying pole convex hulls
    """


    def __init__(self, datafile):
        """
        Initializes a pole classifier

        @param datafile: The datafile name that contains the data
        """
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


    @ignore_warnings(category=ConvergenceWarning)
    def run(self, metric='precision'):
        """
        Featurizes the classifiers data, then using a random training/test split, performs
        grid search to search for the parameters leading to the highest precision/recall/accuracy
        and serializes the model associated to the given metic
        
        @param metric: The metric of the model to serialize
        """

        # Featurize raw data
        X, y = self.featurizer.featurize_for_training(self.data)

        # Different metric scores to search
        scores = ['precision', 'recall', 'accuracy']
        if metric not in scores:
            print("The classifier metric must be one of: ", scores)
            return 
        
        # Params to tune in grid search
        tune_params = [{'penalty': ['l1'], 'C':[0.01, 0.1, 1,10,100,1000], 'dual': [False], 
                         'max_iter': [1000], 'class_weight': ['balanced', None]}, 
                        {'penalty': ['l2'], 'loss': ['hinge', 'squared_hinge'], 'C':[0.01, 0.1,1,10,100,1000], 
                        'max_iter':[1000],'class_weight': ['balanced', None]}]

        # Split the data 80/20 into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, train_size=0.8, shuffle=True)

        best_params = dict()
        for score in scores:
            
            model = GridSearchCV(svm.LinearSVC(), tune_params, score, cv=5, iid=True)
            model.fit(X_train, y_train)

            best_params[score] = model.best_params_

        # Get test results for each score using best parameters
        models = dict()
        for score in scores:
            print("Parameters for highest %s: %s" %(score, best_params[score]))
            model = svm.LinearSVC(**best_params[score])
            model.fit(X_train, y_train)
            models[score] = model
            print("Coefficient vector sparsity: %d/%d" %(len(model.coef_[0]) - np.count_nonzero(model.coef_), len(model.coef_[0])))
            print("Features not used: ", np.argwhere(model.coef_[0] == 0).flatten())
            y_hat_test = model.predict(X_test)
            labels = ["Not Pole", "Pole"]
            print(classification_report(y_test, y_hat_test, target_names=labels))

        # Serialize the model with desired score metric 
        d = os.path.dirname(os.getcwd())
        with open(os.path.join(d, 'pickle/model.pkl'), 'wb') as file:
            pickle.dump(models[metric], file)
            print("Updated model.pkl")

        self.plot(X_test, y_test, models[metric].predict(X_test))



