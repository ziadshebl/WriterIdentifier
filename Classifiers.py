from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier


class Classifiers:
    
    @staticmethod
    def knn_classifier(x_train, y_train, x_test, k=3):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(x_train, y_train)
        results = clf.predict_proba(x_test)
        return results

    @staticmethod
    def svm_classifier(x_train, y_train, x_test):
        clf = svm.SVC(probability=True)  # , gamma='auto', C=5.0)
        clf.fit(x_train, y_train)
        results = clf.predict_proba(x_test)
        return results

    @staticmethod
    def rand_forest_classifier(x_train, y_train, x_test, max_depth=2, random_state=0):
        clf = RandomForestClassifier(
            max_depth=max_depth, random_state=random_state)
        clf.fit(x_train, y_train)
        results = clf.predict_proba(x_test)
        return results

    @staticmethod
    def k_means_classifier(x_train, x_test, n_clusters=3, random_state=0):
        clf = KMeans(n_clusters=n_clusters, random_state=random_state)
        clf.fit(x_train)
        results = clf.predict(x_test)
        results += 1
        return results

    @staticmethod
    def adaboost_classifier(x_train, y_train, x_test, n_estimators=100, random_state=0):
        clf = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state)
        clf.fit(x_train, y_train)
        results = clf.predict_proba(x_test)
        return results

    @staticmethod
    def nn_classifier(x_train, y_train, x_test,  hidden_layer_sizes, max_iter=300, random_state=0):
        clf = MLPClassifier(random_state=random_state, max_iter=max_iter, hidden_layer_sizes=hidden_layer_sizes)
        clf.fit(x_train, y_train)
        results = clf.predict_proba(x_test)
        return results

    @staticmethod
    def gradient_classifier(x_train, y_train, x_test, n_estimators=100, random_state=0, learning_rate=1.0, max_depth=8):
        clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                                         random_state=random_state)
        clf.fit(x_train, y_train)
        results = clf.predict_proba(x_test)
        return results
