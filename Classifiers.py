import cv2
import numpy as np
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from statistics import mode


class Classifiers:
    
    @staticmethod
    def KNNClassifier(x_train, y_train, x_test, k=3):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(x_train, y_train)
        results = clf.predict_proba(x_test)
        return results

    @staticmethod
    def SVMClassifier(x_train, y_train, x_test):
        clf = svm.SVC(probability=True)  # , gamma='auto', C=5.0)
        clf.fit(x_train, y_train)
        results = clf.predict_proba(x_test)
        return results

    @staticmethod
    def RandForestClassifier(x_train, y_train, x_test, max_depth=2, random_state=0):
        clf = RandomForestClassifier(
            max_depth=max_depth, random_state=random_state)
        clf.fit(x_train, y_train)
        results = clf.predict_proba(x_test)
        return results

    @staticmethod
    def KMeansClassifier(x_train, y_train, x_test, n_clusters=3, random_state=0):
        clf = KMeans(n_clusters=n_clusters, random_state=random_state)
        clf.fit(x_train)
        results = clf.predict(x_test)
        results + 1
        return results

    @staticmethod
    def AdaboostClassifier(x_train, y_train, x_test,n_estimators=100, random_state=0):
        clf = AdaBoostClassifier(n_estimators=n_estimators,
                                random_state=random_state)
        clf.fit(x_train, y_train)
        results = clf.predict_proba(x_test)
        return results

    @staticmethod
    def NNClassifier(x_train, y_train, x_test, max_iter=300, random_state=0, hidden_layer_sizes=[100, 50]):
        clf = MLPClassifier(random_state=random_state,
                            max_iter=max_iter, hidden_layer_sizes=hidden_layer_sizes)
        clf.fit(x_train, y_train)
        results = clf.predict_proba(x_test)
        return results
