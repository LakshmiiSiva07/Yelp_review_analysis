import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.svm import *
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import PorterStemmer
import string

#Read data
yelp = pd.read_csv('yelp.csv')
yelp.shape
yelp.head()

# Extracting Train and Test data
X = yelp['text']
y_multi_stars = yelp['stars']
print(y_multi_stars[:5])
print(y_multi_stars.value_counts())

#Converting stars to demonstrate extreme polarity
y = y_multi_stars.copy()
for i in range(0,len(y)):
    if(y[i] > 3):
        y[i] = 1 # Positive
    else:
        y[i] = 0 # Negative
print(y[:5])
print(y.value_counts())

#Preprocessing
def pre_process(text):
    '''
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Changes all string to lower case
    3. Return the list of words after stemming
    '''
    stemmer= PorterStemmer()
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    return [stemmer.stem(word.lower()) for word in nopunc.split() if word.lower() not in stopwords.words('english')]

#Sample pre-processing

sample_text = "Hey there! This is a sample review, which happens or happening to contain or CONTAINS punctuations."
print(pre_process(sample_text)) # ['hey', 'sampl', 'review', 'happen', 'happen', 'contain', 'contain' 'punctuat']

#Bag of words Feauture Extraction
bow_transformer = CountVectorizer(analyzer=pre_process).fit(X)
X = bow_transformer.transform(X)

#Splitting dataset by bag of words feature
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

#Training and Fitting SVM Model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train,y_train)
preds = svm_model.predict(X_test)
print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds))
print(accuracy_score(y_test,preds)) #0.78

