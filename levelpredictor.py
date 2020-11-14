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


y_pred= mod.regressor.predict(mod.X_test)
#This gives us the predicted values for our test set.
#This snippet helps to obtain them in a parsed format.

def predictor():
    y_pred_mod = []
    y_test_mod = []

    for i in range(0, 4):
        j = 0
        y_pred_temp = []
        y_test_temp = []

        while (j < len(mod.y_test)):
            y_pred_temp.append(y_pred[j][i])
            y_test_temp.append(mod.y_test[j][i])
            j += 1

        y_pred_mod.append(np.array(y_pred_temp))
        y_test_mod.append(np.array(y_test_temp))

        #Writing to text files for data reference check
        with open("y_pred_mod.txt", "w") as text_file:
            print(f" {y_pred_mod}", file=text_file)
        with open("y_test_mod.txt", 'w') as text_file:
            print(f" {y_test_mod}", file=text_file)
        with open("y_pred_temp.txt", 'w') as text_file:
            print(f" {y_pred_temp}", file=text_file)
        with open("y_test_temp.txt", 'w') as text_file:
            print(f" {y_test_temp}", file=text_file)

        #print(len(mod.y_test)) #Returns 817

#This gives us the predicted values for our test set. This snippet helps to obtain them in a parsed format.
    #High
    fig, ax = plt.subplots()
    ax.scatter(y_test_mod[0], y_pred_mod[0])
    ax.plot([y_test_mod[0].min(), y_test_mod[0].max()], [y_test_mod[0].min(), y_test_mod[0].max()], 'k--', lw=4)
    plt.title('High')
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()
    #Low
    fig, ax = plt.subplots()
    ax.scatter(y_test_mod[1], y_pred_mod[1])
    ax.plot([y_test_mod[1].min(), y_test_mod[1].max()], [y_test_mod[1].min(), y_test_mod[1].max()], 'k--', lw=4)
    plt.title('Low')
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()
    #Open
    fig, ax = plt.subplots()
    ax.scatter(y_test_mod[2], y_pred_mod[2])
    ax.plot([y_test_mod[2].min(), y_test_mod[2].max()], [y_test_mod[2].min(), y_test_mod[2].max()], 'k--', lw=4)
    plt.title('Open')
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()
    #Close
    fig, ax = plt.subplots()
    ax.scatter(y_test_mod[3], y_pred_mod[3])
    ax.plot([y_test_mod[3].min(), y_test_mod[3].max()], [y_test_mod[3].min(), y_test_mod[3].max()], 'k--', lw=4)
    plt.title('Close')
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()

#Now, we can see our model has performed quite well considering most of the points lie on the regression line and
# there are very few outliers.

predictor()

### Regression completed

