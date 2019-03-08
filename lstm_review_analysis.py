import pandas as pd
import numpy as np
import matplotlib.pyplot as pltr
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import keras 
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential

yelp = pd.read_csv('yelp.csv')
yelp.shape
yelp.head()

X = yelp['text']
y_multi_stars = yelp['stars']
print(y_multi_stars[:5])
print(y_multi_stars.value_counts())

tokenizer = Tokenizer(num_words=2500,split=' ')
tokenizer.fit_on_texts(X)

X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X)

y = y_multi_stars.copy()
for i in range(0,len(y)):
    if(y[i] > 3):
        y[i] = 1
    else:
        y[i] = 0
print(y[:5])
print(y.value_counts())

y= to_categorical(y)
print(y)

x_train, x_test, y_train, y_test = train_test_split(X,y,train_size=0.8)

model = Sequential()
model.add(Embedding(2500,128,input_length=X.shape[1],dropout=0.2))
model.add(LSTM(300, dropout_U=0.2,dropout_W=0.2))
model.add((Dense(2,activation = 'softmax')))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])
print(model.summary())
model.fit(x_train,y_train,epochs=10,verbose=2,batch_size=32)
print(model.evaluate(x_test,y_test)[1])
