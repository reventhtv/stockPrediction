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

df = pd.read_csv('stock_details/AMZN.csv', index_col=0, parse_dates=True)

def plotdata():
    # This snippet will help us to pick the Adjusted Close column of each stock other than our target stock
    # which is AMZN, rename the column as the ticker and merge it in our feature set.
    # It will produce a feature set like this. The Date is the index and corresponding to the Date,
    # each ticker’s “Adj Close” value. Now, We will see there are a few empty columns initially.
    # This is because these companies didn’t start to participate in the stock market back in 2010.
    # This will give us a feature set of 200 columns containing 199 company’s values and the Date.

    # Now, let’s focus on our target stock the AMZN stock.
    # we will start visualizing each of our given column values for the target stock.

    # Now, let’s visualize, our stock using the candlestick notation. I am using Pandas version 0.24.2 for this.
    # There may be an issue as in the current versions this module is depreciated.


    df_ohlc = df['Adj Close'].resample('10D').ohlc()
    # print(df_ohlc.head())
    df_volume = df['Volume'].resample('10D').sum()
    df_ohlc.reset_index(inplace=True)
    df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)
    ax1.xaxis_date()
    candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
    ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
    plt.show()

plotdata()


def featuredata():
    # Now, let’s devise some features that will help us to predict our target.
    # We will calculate the 50 moving average.
    # This characteristic is used by a lot of traders for predictions.
    # New column "Moving_av" is added to the dataframe
    df['Moving_av'] = df['Adj Close'].rolling(window=50, min_periods=0).mean()
    # print(df.head())
    df['Moving_av'].plot()

    # Now, we will try to obtain two more features, Rate of increase in volume and rate of increase in Adjusted Close for our stock

    i = 1
    rate_increase_in_vol = [0]
    rate_increase_in_adj_close = [0]
    while i < len(df):
        rate_increase_in_vol.append(df.iloc[i]['Volume'] - df.iloc[i - 1]['Volume'])
        rate_increase_in_adj_close.append(df.iloc[i]['Adj Close'] - df.iloc[i - 1]['Adj Close'])
        i += 1

    df['Increase_in_vol'] = rate_increase_in_vol
    df['Increase_in_adj_close'] = rate_increase_in_adj_close
    df.to_csv("dataset_target_2.csv", index=False)
    # print(df.head())
    # df['Increase_in_vol'].plot()
    # df['Increase_in_adj_close'].plot()



def mergedata():
    featuredata()
    # Now, our feature file for our target stock is ready.
    # Now, we merge both these feature files to make the main feature set.


    df1 = pd.read_csv('dataset_target_2.csv')

    df3 = pd.read_csv('stock_details/AMZN.csv')
    df2 = pd.read_csv('Dataset_temp.csv')

    Dates = []
    i = 0
    while i < len(df3):
        Dates.append(df3.iloc[i]['Date'])
        i += 1

    df_new = df1.join(df2, how='outer')
    df_new.fillna(0.0)

    df_new['Date'] = Dates

    df_new.to_csv('Dataset_main.csv', index=False)

    # print(df2.head())
    # print(df_new.head())

mergedata()