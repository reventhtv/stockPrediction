import bs4 as bs
import pickle
import requests
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web

import matplotlib.pyplot as plt
from matplotlib import style
#from mplfinance import candlestick_ohlc
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import seaborn as sb
import numpy as np
import model as mod

from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

#Recurrent Neural Network
# We will be using RNN for the past analysis for our target stock. So, we will be working with the target stock feature file only.
#Here we will use LSTM layers that work on RNN principles but work on a gated approach
#LSTM helps the RNN to keep the memory over a long period of time, solving the vanishing and exploding gradient problems.
#Reference: https://colah.github.io/posts/2015-08-Understanding-LSTMs/


df1 = pd.read_csv('stock_details/AMZN.csv')  ##AMZN is our target stock. Another way is to store target stock in a new dataset (Future)

print(df1.head())

df2 = pd.read_csv('dataset_target_2.csv')  # df2 is out dataset for LSTM

print(df2.head())

# We dont need column 'Date' for our dataset df2 because it has no correlation.
# If you see drop it: ->df2.drop(['Date'], axis=1)
# We need to predict High, Low, Open, Close. we will use all 9 columns in df2 dataframe as our input values
# correspondingly our output will have 4 columns High, Low, Open, Close of our target stock
# We need to create the train and test set for our LST model. For this purpose we will split our data of length 2721 records
# into two parts. we will be using 0-2200 records as the train set and 2200-2721 as our test set.
# we will select the target set comprising of the 4 target columns from our train set
print(len(df2))  # returns 2721

df_train = df2[:1000]

print(df_train.head())

# Now we will scale our data to make our model converge easily, as we have large variations of values in our data set
sc = MinMaxScaler(feature_range=(0, 1))

df_target = df_train[['High', 'Low', 'Open', 'Close']]

target_set = df_target.values  # df_target has only 4 columns 'High', 'Low', 'Open', 'Close'
train_set = df_train.values  # df_train has 9 columns

training_set_scaled = sc.fit_transform(train_set)
target_set_scaled = sc.fit_transform(target_set)
# We have obtained the scaled data for our LSTM model
# LSTM model takes in series data and produces output. Out LSTM is many to many RNN model.
# So, we need to produce a series of data for this purpose. To do this, we have started from the 50th index and
# move to the length of the training set. We have appended 0-49, which is 50 values to a list.
# we have created such lists for all our features.

X_train = []
y_train = []
for i in range(50, len(train_set)):
    X_train.append(training_set_scaled[i - 50:i, :])
    y_train.append(target_set_scaled[i, :])

X_train, y_train = np.array(X_train), np.array(y_train)

print(X_train.shape)  # returns (2150, 50, 9)
print(y_train.shape)  # returns (2150, 4)

# So our input has (n x 50 x 9) dimensional data for our training set. we have 9 features and each feature is a list of the
# feature values for an extended period of 50 days. n is the number of such series obtained from the given dataset.
# Now our target set is the value of the target columns on the 51st day

from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import BatchNormalization


def model():
    #This is our LSTM model. we have used keras laters here. The loss function is mean squared error. we have used 'Adam Optimizer'
    mod = Sequential()
    mod.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 9)))
    mod.add(Dropout(0.2))
    mod.add(BatchNormalization())
    mod.add(LSTM(units=64, return_sequences=True))
    mod.add(Dropout(0.1))
    mod.add(BatchNormalization())

    mod.add((LSTM(units=64)))
    mod.add(Dropout(0.1))
    mod.add(BatchNormalization())
    mod.add((Dense(units=16, activation='tanh')))
    mod.add(BatchNormalization())
    mod.add((Dense(units=4, activation='tanh')))
    mod.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', 'mean_squared_error'])
    mod.summary()

    return mod


RNN_model=model()

#Now we will train our model by using below snippet

import tensorflow as tf
callback=tf.keras.callbacks.ModelCheckpoint(filepath='./RNN_model.h5',
                                           monitor='mean_squared_error',
                                           verbose=0,
                                           save_best_only=True,
                                           save_weights_only=False,
                                           mode='auto',
                                           save_freq='epoch')
RNN_model.fit(X_train, y_train, epochs = 2000, batch_size = 32,callbacks=[callback])

#Our model is trained now. Let's focus on testing part now
#We have 2200 to 2636 records for our test values.
# So, we obtain our target values by picking the four columns Open, Close, High, Low of our stock

df_test = df2[1000:]
df_target_test = df_test[['High', 'Low', 'Open', 'Close']]
target_set_test = df_target_test.values
test_set = df_test.values

test_set_scaled = sc.fit_transform(test_set)
target_set_scaled = sc.fit_transform(target_set_test)

X_test = []
y_test = []
for i in range(50,len(test_set)):
    X_test.append(test_set_scaled[i-50:i,:])
    y_test.append(target_set_scaled[i,:])

#To test also, we need to transform our test feature dataset and form a series of 50 feature values for this set as we did in case of the training set above
X_test, y_test = np.array(X_test), np.array(y_test)
predicted_stock_price = RNN_model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

print(predicted_stock_price)

#High
plt.figure(figsize=(20,10))
plt.plot(target_set_test[0], color = 'green', label = 'Real Amazon stock')
plt.plot(predicted_stock_price[0], color = 'red', label = 'Predicted Amazon Stock Price')
plt.title('Amazon Stock Price Prediction')
plt.xlabel('Trading Day')
plt.ylabel('Amazon Stock Price')
plt.legend()
plt.show()

#Low
plt.figure(figsize=(20,10))
plt.plot(target_set_test[1], color = 'green', label = 'Real Amazon stock')
plt.plot(predicted_stock_price[1], color = 'red', label = 'Predicted Amazon Stock Price')
plt.title('Amazon Stock Price Prediction')
plt.xlabel('Trading Day')
plt.ylabel('Amazon Stock Price')
plt.legend()
plt.show()

#Open
plt.figure(figsize=(20,10))
plt.plot(target_set_test[2], color = 'green', label = 'Real Amazon stock')
plt.plot(predicted_stock_price[2], color = 'red', label = 'Predicted Amazon Stock Price')
plt.title('Amazon Stock Price Prediction')
plt.xlabel('Trading Day')
plt.ylabel('Amazon Stock Price')
plt.legend()
plt.show()

#Close
plt.figure(figsize=(20,10))
plt.plot(target_set_test[3], color = 'green', label = 'Real Amazon stock')
plt.plot(predicted_stock_price[3], color = 'red', label = 'Predicted Amazon Stock Price')
plt.title('Amazon Stock Price Prediction')
plt.xlabel('Trading Day')
plt.ylabel('Amazon Stock Price')
plt.legend()
plt.show()

#All over
plt.figure(figsize=(20,10))
plt.plot(target_set_test, color = 'green', label = 'Real Amazon stock')
plt.plot(predicted_stock_price, color = 'red', label = 'Predicted Amazon Stock Price')
plt.title('Amazon Stock Price Prediction')
plt.xlabel('Trading Day')
plt.ylabel('Amazon Stock Price')
plt.legend()
plt.show()
