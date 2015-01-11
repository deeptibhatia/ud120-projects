#!/usr/bin/python

#lesson 11 
import pickle
import numpy
import operator

numpy.random.seed(42)


### the words (features) and authors (labels), already largely processed
#words_file = "word_data_overfit.pkl" ### like the file you made in the last mini-project
#authors_file = "email_authors_overfit.pkl"  ### this too
words_file = "../text_learning/your_word_data.pkl" ### like the file you made in the last mini-project
authors_file = "../text_learning/your_email_authors.pkl"  ### this too

word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (remainder go into training)
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train).toarray()
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150]
labels_train   = labels_train[:150]



### your code goes here

print "Q: How many number of features?"
print "A: ", len(features_train)

from sklearn import tree
from sklearn.metrics import accuracy_score

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(labels_test, pred)

print "Q: What's the accuracy of the decision tree?"
print "A: ", acc

fis = clf.feature_importances_
print len(fis)

print "Relative importance of features"
fis_imp =  [(i, fi) for i, fi in enumerate(fis) if float(fi) > 0.02]



most_imp_feature = max(fis_imp, key=operator.itemgetter(1))

names = vectorizer.get_feature_names()
print len(names)
print most_imp_feature, names[most_imp_feature[0]]
#print [(i, name) for i, name in enumerate(names)]

print len(fis), len(fis_imp)
print fis_imp

##Confirming index
print numpy.where(fis>0.76)
for i, fi in enumerate(fis):
    if fi > 0.76:
        print i, fi
