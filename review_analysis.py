import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

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
