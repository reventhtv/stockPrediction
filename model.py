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
import tensorflow as tf
#from dataprocessing import processdata
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

#ML/DL Algorithm modelling
#Splitting our sets into training adn testing
from sklearn.model_selection import train_test_split

readX = pd.read_csv("X.csv")
readY = pd.read_csv("y.csv")

X = readX.values
y = readY.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape, X_train)
print(y_train.shape, y_train)

def model():
    mod = Sequential()
    mod.add(Dense(32, kernel_initializer='normal', input_dim=200, activation='relu'))
    mod.add(Dense(64, kernel_initializer='normal', activation='relu'))
    mod.add(Dense(128, kernel_initializer='normal', activation='relu'))
    mod.add(Dense(256, kernel_initializer='normal', activation='relu'))
    mod.add(Dense(4, kernel_initializer='normal', activation='linear'))

    mod.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy', 'mean_absolute_error'])
    mod.summary()

    return mod


regressor = KerasRegressor(build_fn=model, batch_size=16, epochs=2000)

# I have used Keras Regressor for this purpose. Saved best weights and used 2000 epochs. Mean Absolute Error is our Loss function.
# We have an input of 200 after dropping the columns. Next, We are going to obtain four values for each input, High, Low, Open, Close.
callback = tf.keras.callbacks.ModelCheckpoint(filepath='Regressor_model.h5',
                                              monitor='mean_absolute_error',
                                              verbose=0,
                                              save_best_only=True,
                                              save_weights_only=False,
                                              mode='auto')
results = regressor.fit(X_train, y_train, callbacks=[callback])
#print(results)

