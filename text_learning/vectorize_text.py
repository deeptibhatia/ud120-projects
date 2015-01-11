#!/usr/bin/python

import pickle
import sys
import re
sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText

"""
    starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification

    the list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    the actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project

    the data is stored in lists and packed away in pickle files at the end

"""


from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list
temp_counter = 0


for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        temp_counter += 1
        if temp_counter < 200:
            path = "../"+path[:-1]
            print path
            email = open(path, "r")

            ### use parseOutText to extract the text from the opened email
            text_string = parseOutText(email)

            ### use str.replace() to remove any instances of the words
            ### ["sara", "shackleton", "chris", "germani"]
            for word in text_string.split():
              if word.strip() in {"sara", "shackleton", "chris", "germani"}:
                text_string = text_string.replace(word, "")

            ### append the text to word_data
            word_data.append(text_string)

            ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
            if name == 'sara':
               from_data.append(0) 
            elif name == 'chris':
               from_data.append(1)


            email.close()

print "emails processed"
from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data.pkl", "w") )
pickle.dump( from_data, open("your_email_authors.pkl", "w") )

### in Part 4, do TfIdf vectorization here
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
sw = stopwords.words("english")

vec = TfidfVectorizer(stop_words=sw)
train_data_features = vec.fit_transform(word_data)
print "TFIDF***"
names = vec.get_feature_names()
print len(names)
print names[34597]

#meaningful_list = []
#for text in word_data:
#      meaningful_words = [w.strip() for w in text.split() if not w in sw]
#          meaningful_list.append(" ".join(meaningful_words))
#
#vec2 = TfidfVectorizer()
#train_data_features2 = vec2.fit_transform(meaningful_list)
#names2 = vec2.get_feature_names()
#idf = vec2._tfidf.idf_
#print "Filtered words data", len(names2)

#Had to convert the data to unix format using the command: find . -type f -exec dos2unix {} \;
