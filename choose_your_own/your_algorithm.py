#!/usr/bin/python

#lesson 4
import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
#################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn import tree

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def adaboost():
    return ("AdaBoost", AdaBoostClassifier(n_estimators=50, learning_rate=0.5, algorithm = 'SAMME'))

def random_forest():
    return ("RandomForest", RandomForestClassifier(n_estimators=100))

def decision_tree():
    return ("DecisionTree", tree.DecisionTreeClassifier(min_samples_split=40))

def gaussian():
    return ("Gaussian", GaussianNB())

def svm():
    #return SVC(kernel = 'rbf', C=0.9, gamma = 10.0)  #Gives accuracy of 0.936
    return ("SVC", SVC(kernel = 'rbf', C=0.85, gamma = 11.0, coef0 = 5, tol=0.00000001, probability=True))

def kneighbors():
    return ("K Neighbors BaggingClassifier", BaggingClassifier(KNeighborsClassifier(),
                            max_samples=0.6))

def getClassifier(k):
    options = { 0: adaboost,
                1: random_forest,
                2: decision_tree,
                3: gaussian,
                4: svm
    }
    try:
        clf = options[k]()
        return clf
    except:
        return kneighbors()

def accuracy_scores():
    accuracy_scores = []
    for idx in range(0,6):
        name, classifier = getClassifier(idx)
        classifier = classifier.fit(features_train, labels_train)
        labels_pred = classifier.predict(features_test)
        accuracy_scores.append((idx, accuracy_score(labels_test, labels_pred)))
    return accuracy_scores

def max_accuracy_classifier():
    import operator
    accuracy_algorithms = accuracy_scores()
    print accuracy_algorithms
    return max((accuracy_algorithms), key=operator.itemgetter(1))

print "Features count: ",  len(features_train[0])
max_acc_idx, max_acc_score = max_accuracy_classifier()
max_acc_name, max_acc_clf =  getClassifier(max_acc_idx)
print "Max accuracy classifier", max_acc_name, max_acc_score

try:
    max_acc_clf.fit(features_train, labels_train)
    print "Creating pretty picture"
    prettyPicture(max_acc_clf, features_test, labels_test)
except NameError:
    pass
