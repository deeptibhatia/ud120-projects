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

def getClassifier(k):
    if k == 0:
        print "Choosing Adaboost classifier"
        return AdaBoostClassifier(n_estimators=50, learning_rate=0.5, algorithm = 'SAMME')
    elif k == 1:
        print "Choosing Random Forest classifier"
        return RandomForestClassifier(n_estimators=100)
    elif k == 2:
        print "Choosing Decision Tree"
        return tree.DecisionTreeClassifier(min_samples_split=40)
    elif k == 3:
        print "Choosing Gaussian Naive Bayes"
        return GaussianNB()
    elif k == 4:
        print "Choosing SVM"
        #return SVC(kernel = 'rbf', C=0.9, gamma = 10.0)  #Gives accuracy of 0.936
        return SVC(kernel = 'rbf', C=0.85, gamma = 11.0, coef0 = 5, tol=0.00000001, probability=True)
    else:
        print "Choosing K Neighbors classifier"
        return BaggingClassifier(KNeighborsClassifier(),
                            max_samples=0.6)

print "Features: ",  len(features_train[0])

import sys
classifier_type = 0
if len(sys.argv) > 1:
    classifier_type = int(sys.argv[1])
else:
    print "Choosing default classifier type\n"
clf = getClassifier(classifier_type)

clf = clf.fit(features_train, labels_train)
labels_pred = clf.predict(features_test)

acc = accuracy_score(labels_test, labels_pred)

#scores = cross_val_score(clf, features_train, labels_train)
#mean = scores.mean()
print "Acc: ", acc

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
