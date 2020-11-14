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

# we will be using the S&P 500 companies database
# I have collected the S&P list from the Wikipedia page using web scrapping
# I am using this as a source because it is updated live I think
# This code will help you to scrap the tickers of the top 500 companies


def save_tickers():

  resp=requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
  soup=bs.BeautifulSoup(resp.text, features="lxml")
  table=soup.find('table',{'class':'wikitable sortable'})
  tickers=[]
  for row in table.findAll('tr')[1:]:
   ticker=row.findAll('td')[0].text[:-1]
   tickers.append(ticker)

  with open("tickers.pickle",'wb') as f:
   pickle.dump(tickers, f)

   print(tickers)
   return tickers


save_tickers()



# I have then collected their details from Yahoo using the Pandas web-data reader.

def fetch_data():
    with open("tickers.pickle", 'rb') as f:
        tickers = pickle.load(f)

    if not os.path.exists('stock_details'):
      os.makedirs('stock_details')
    count = 200

    start = dt.datetime(2010, 1, 1)
    end = dt.datetime(2020, 10, 22)
    count = 0
    for ticker in tickers:
        if count == 200:
            break
        count += 1
        print(ticker)

        try:
            df = web.DataReader(ticker, 'yahoo', start, end)
            df.to_csv('stock_details/{}.csv'.format(ticker))
        except:
            print("Error")
            continue


fetch_data()

#We kept a count of 200 because I wanted to use only 200 company details. In these 200 companies,
# #we will have a target company and 199 companies that will help to reach a prediction about our target company.

#This code will generate a ‘stock_details’ folder which will have 200 company details from 1st January 2010 to 22nd June 2020.




#Each detail file will be saved by its stock’s ticker. I have chosen Amazon as my target stock.
# So, I will try to predict the stock prices for Amazon. Its ticker is AMZN.
# It has ‘Date’ as the index feature. ‘High’ denotes the highest value of the day.
# ‘Low’ denotes the lowest. ‘Open’ is the opening Price and ‘Close’ is the closing for that Date.
# Now, sometimes close values are regulated by the companies.
# So the final value is the ‘Adj Close’ which is the same as ‘Close’ Value if the stock price is not regulated.
# ‘Volume’ is the amount of Stock of that company traded on that date.
#We will consider this ‘Adj Close’ value of each stock as the contributing feature of each stock on our target stock.
# So, we will rename the Adj Close of each stock as the corresponding stock ticker and include it in our feature set.


def compile():
    with open("tickers.pickle", 'rb') as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        if 'AMZN' in ticker:
            continue
        if not os.path.exists('stock_details/{}.csv'.format(ticker)):
            continue
        df = pd.read_csv('stock_details/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        df.rename(columns={'Adj Close': ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', "Close", 'Volume'], axis=1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

    #print(main_df.head())
    main_df.to_csv('Dataset_temp.csv')


compile()

#This snippet will help us to pick the Adjusted Close column of each stock other than our target stock
# which is AMZN, rename the column as the ticker and merge it in our feature set.

#It will produce a feature set like this. The Date is the index and corresponding to the Date,
# each ticker’s “Adj Close” value. Now, We will see there are a few empty columns initially.
# This is because these companies didn’t start to participate in the stock market back in 2010.
# This will give us a feature set of 200 columns containing 199 company’s values and the Date.

#Now, let’s focus on our target stock the AMZN stock.
#we will start visualizing each of our given column values for the target stock.

#Now, let’s visualize, our stock using the candlestick notation. I am using Pandas version 0.24.2 for this.
# There may be an issue as in the current versions this module is depreciated.


df=pd.read_csv('stock_details/AMZN.csv',index_col=0,parse_dates=True)
df_ohlc= df['Adj Close'].resample('10D').ohlc()
#print(df_ohlc.head())
df_volume=df['Volume'].resample('10D').sum()
df_ohlc.reset_index(inplace=True)
df_ohlc['Date']=df_ohlc['Date'].map(mdates.date2num)
ax1=plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2=plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1 , sharex=ax1)
ax1.xaxis_date()
candlestick_ohlc(ax1,df_ohlc.values, width=2, colorup='g')
ax2.fill_between(df_volume.index.map(mdates.date2num),df_volume.values,0)
plt.show()



#Now, let’s devise some features that will help us to predict our target.
#We will calculate the 50 moving average.
# This characteristic is used by a lot of traders for predictions.
#New column "Moving_av" is added to the dataframe
df['Moving_av']= df['Adj Close'].rolling(window=50,min_periods=0).mean()
#print(df.head())
df['Moving_av'].plot()


#Now, we will try to obtain two more features, Rate of increase in volume and rate of increase in Adjusted Close for our stock

i = 1
rate_increase_in_vol = [0]
rate_increase_in_adj_close = [0]
while i < len(df):
    rate_increase_in_vol.append(df.iloc[i]['Volume'] - df.iloc[i - 1]['Volume'])
    rate_increase_in_adj_close.append(df.iloc[i]['Adj Close'] - df.iloc[i - 1]['Adj Close'])
    i += 1

df['Increase_in_vol'] = rate_increase_in_vol
df['Increase_in_adj_close'] = rate_increase_in_adj_close
df.to_csv("dataset_target_2.csv",index=False)
#print(df.head())
#df['Increase_in_vol'].plot()
#df['Increase_in_adj_close'].plot()
#Now, our feature file for our target stock is ready.
#Now, we merge both these feature files to make the main feature set.

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

#print(df2.head())
#print(df_new.head())





#DATA PROCESSING
#Our dataset has 507 columns 500 from the other companies list and 7 from our target (AMZN.csv) stock feature set anchored on the date column.
# So, we have 507 columns

#Regression:
#In this part, we will study the impact of other stocks on our target stock (AMZN.csv)
#We will try to predict the High, Low, Open and Close of our amazon stock. First, lets analyze our data
df1=pd.read_csv('stock_details/AMZN.csv')
col1=df1.columns
#print(col1)

df2=pd.read_csv('dataset_target_2.csv')
col2=df2.columns
#print(col2)

df3=pd.read_csv('Dataset_main.csv')
col3=df3.columns
#print(col3)
#print(df3.head())

#df4=pd.read_csv('Dataset_temp.csv')
##col4=df4.columns
#print(col4)

#using seaborn module here
#Correlation plot for regression
C_mat = df3.corr()
fig = plt.figure(figsize = (15,15))
sb.heatmap(C_mat, vmax = .8, square = True)
plt.show()

#Histogram for regression
df3.hist(figsize = (35,35))
plt.show()

#So from our above plots it is evident that we will have some correlations in our dataset.
# Next, we  we drop the open, close, high, low values from our training dataset and
# use them as target label or values set.
# We will also drop volume and date because they wont have a correlation.

df3.fillna(0, inplace=True)
y_df=df3[['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']]
col_y = y_df.columns
#print(col_y) #returns High, Low, Open, Close, Volume, Adj Close
#print(y_df)

#Dropping 'Adj Close' and 'Volume' columns from dataset
y_df_mod=y_df.drop(['Adj Close', 'Volume'], axis=1)
#print(y_df_mod.columns)


Drop_cols=col_y
Drop_cols=Drop_cols.tolist()
Drop_cols.append('Date') #includes 'Date' column to the dataset Drop_cols= col_y

X_df=df3.drop(Drop_cols,axis=1)
#print(Drop_cols) #returns ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close', 'Date']
#print(X_df) #Returns dataset after dropping 'Drop_cols'
#print(X_df.columns)
X_df.to_csv("X.csv")
#Only the values in the DataFrame will be returned, the axes labels will be removed.
X=X_df.values
print(X)

y=y_df_mod.values
y_df_mod.to_csv("y.csv")
print(y)
#ML/DL Algorithm modelling
#Splitting our sets into training adn testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)
print(X_train.shape, X_train)
print(y_train.shape, y_train)
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score


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

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
regressor = KerasRegressor(build_fn=model, batch_size=16,epochs=2000)

import tensorflow as tf
#print(tf)
#I have used Keras Regressor for this purpose. Saved best weights and used 2000 epochs. Mean Absolute Error is our Loss function.
# We have an input of 200 after dropping the columns. Next, We are going to obtain four values for each input, High, Low, Open, Close.
callback=tf.keras.callbacks.ModelCheckpoint(filepath='Regressor_model.h5',
                                           monitor='mean_absolute_error',
                                           verbose=0,
                                           save_best_only=True,
                                           save_weights_only=False,
                                           mode='auto')
results=regressor.fit(X_train,y_train,callbacks=[callback])




y_pred= regressor.predict(X_test)
print(y_pred)

print(y_test)


import numpy as np
y_pred_mod=[]
y_test_mod=[]

for i in range(0,4):
    j=0
    y_pred_temp=[]
    y_test_temp=[]

    while(j<len(y_test)):
        y_pred_temp.append(y_pred[j][i])
        y_test_temp.append(y_test[j][i])
        j+=1

    y_pred_mod.append(np.array(y_pred_temp))
    y_test_mod.append(np.array(y_test_temp))

