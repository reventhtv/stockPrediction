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