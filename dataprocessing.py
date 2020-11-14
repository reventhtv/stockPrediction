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

def processdata():
    # DATA PROCESSING
    # Our dataset has 507 columns 500 from the other companies list and 7 from our target (AMZN.csv) stock feature set anchored on the date column.
    # So, we have 507 columns

    # Regression:
    # In this part, we will study the impact of other stocks on our target stock (AMZN.csv)
    # We will try to predict the High, Low, Open and Close of our amazon stock. First, lets analyze our data
    df1 = pd.read_csv('stock_details/AMZN.csv')
    col1 = df1.columns
    # print(col1)

    df2 = pd.read_csv('dataset_target_2.csv')
    col2 = df2.columns
    # print(col2)

    df3 = pd.read_csv('Dataset_main.csv')
    col3 = df3.columns
    # print(col3)
    # print(df3.head())

    # df4=pd.read_csv('Dataset_temp.csv')
    ##col4=df4.columns
    # print(col4)

    # using seaborn module here
    # Correlation plot for regression
    C_mat = df3.corr()
    fig = plt.figure(figsize=(15, 15))
    sb.heatmap(C_mat, vmax=.8, square=True)
    plt.show()

    # Histogram for regression
    df3.hist(figsize=(35, 35))
    plt.show()

    # So from our above plots it is evident that we will have some correlations in our dataset.

    # Next, we  we drop the open, close, high, low values from our training dataset and
    # use them as target label or values set.
    # We will also drop volume and date because they wont have a correlation.

    df3.fillna(0, inplace=True)
    y_df = df3[['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']]
    col_y = y_df.columns
    # print(col_y) #returns High, Low, Open, Close, Volume, Adj Close
    # print(y_df)

    # Dropping 'Adj Close' and 'Volume' columns from dataset
    y_df_mod = y_df.drop(['Adj Close', 'Volume'], axis=1)
    # print(y_df_mod.columns)

    Drop_cols = col_y
    Drop_cols = Drop_cols.tolist()
    Drop_cols.append('Date')  # includes 'Date' column to the dataset Drop_cols= col_y

    X_df = df3.drop(Drop_cols, axis=1)
    # print(Drop_cols) #returns ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close', 'Date']
    # print(X_df) #Returns dataset after dropping 'Drop_cols'
    # print(X_df.columns)
    X_df.to_csv("X.csv", index=False)
    # Only the values in the DataFrame will be returned, the axes labels will be removed.
    X = X_df.values
    print(X)

    y = y_df_mod.values
    y_df_mod.to_csv("y.csv", index=False)
    print(y)

processdata()