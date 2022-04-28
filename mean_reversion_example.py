from calendar import c
from datetime import date
from turtle import color
from numpy import diff
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np


# local imports
from backtester import engine, tester
from backtester import API_Interface as api

training_period = 119 # How far the rolling average takes into calculation

'''
logic() function:
    Context: Called for every row in the input data.
    Input:  account - the account object
            lookback - the lookback dataframe, containing all data up until this point in time
    Output: none, but the account object will be modified on each call
'''
def logic(account, lookback): # Logic function to be used for each time interval in backtest 
    
    # Commonly used
    today = len(lookback)-1
    price = lookback['close'][today]
     
    # If the position for today is 1 meaning there is a bullish crossover we look for closing any existing shorts and longing
    if(lookback['position'][today] == 1):
        for position in account.positions: # Close all current positions
            account.close_position(position, 1, lookback['close'][today])
        
        if(account.buying_power > 0):
            account.enter_position('long', account.buying_power, price)

    # If the position for today is -1 meaning there is a bearish crossover we look for closing any existing longs and shorting
    if(lookback['position'][today] == -1):
        
        for position in account.positions: # Close all current positions
            account.close_position(position, 1, lookback['close'][today])
        
        if(account.buying_power > 0):
            account.enter_position('short', account.buying_power, price)
            

'''
preprocess_data() function:
    Context: Called once at the beginning of the backtest. TOTALLY OPTIONAL. 
             Each of these can be calculated at each time interval, however this is likely slower.
    Input:  list_of_stocks - a list of stock data csvs to be processed
    Output: list_of_stocks_processed - a list of processed stock data csvs
'''
def preprocess_data(list_of_stocks):
    list_of_stocks_processed = []
    for stock in list_of_stocks:
        df = pd.read_csv("original_data/" + stock + ".csv", parse_dates=[0])
        df = df.iloc[::60, :]
        
        # Best Parameters is EMA FAST at 20 and EMA SLOW at 119 for average of about 90 percent return
        df['EMAFAST'] = df['close'].ewm(span = (20*12), adjust=False, min_periods=1).mean()
        df['EMASLOW'] = df['close'].ewm(span = (119*24), adjust=False, min_periods=1).mean()
        
        # Long signal where the fast crosses above the slow and position shows whether we long or short by taking difference between our long signals
        # There will be a 1 for a bullish crossover where the EMAFAST crosses above the EMASLOW and a -1 for a bearish crossover where the EMASFAST crosses below the EMASLOW
        # https://towardsdatascience.com/making-a-trade-call-using-simple-moving-average-sma-crossover-strategy-python-implementation-29963326da7a
        
        df['longsignal'] = np.where(df['EMAFAST'] > df['EMASLOW'], 1.0, 0.0)
        df['position'] = df['longsignal'].diff()

        
        # Cool plots of our strategy
        # plt.figure(figsize = (20,10))
        # # plot close price, short-term and long-term moving averages 
        # df['close'].plot(color = 'k', label = 'Close Price')
        # df['EMAFAST'].plot(color = 'b',label = '20-day EMA'); 
        # df['EMASLOW'].plot(color = 'g', label = '119-day EMA');
        
        # # plot ‘buy' signals
        # plt.plot(df[df['position'] == 1].index, 
        #         df['EMAFAST'][df['position'] == 1], 
        #         '^', markersize = 15, color = 'g', label = 'buy')
        # # plot ‘sell' signals
        # plt.plot(df[df['position'] == -1].index, 
        #         df['EMAFAST'][df['position'] == -1], 
        #         'v', markersize = 15, color = 'r', label = 'sell')

        # plt.ylabel('Price', fontsize = 15 )
        # plt.xlabel('Date', fontsize = 15 )
        # plt.title(stock, fontsize = 20)
        # plt.legend()
        # plt.grid()
        # plt.show()
        list_of_stocks_processed.append(stock + "_Processed")
        df.to_csv("data/" + stock + "_Processed.csv", index=False) # Save to CSV
        
        
    return list_of_stocks_processed


if __name__ == "__main__":
    # list_of_stocks = ["TSLA_2020-03-01_2022-01-20_1min"] 
    list_of_stocks = ["DIS", "AMZN", "JPM", "GOOG", "PEP", "NVDA", "FB", "JNJ", "V", "UNH", "MSFT", "LLY", "KO"] # List of stock data csv's to be tested, located in "data/" folder 
    list_of_stocks_proccessed = preprocess_data(list_of_stocks) # Preprocess the data
    results = tester.test_array(list_of_stocks_proccessed, logic, chart=True) # Run backtest on list of stocks using the logic function

    # print("training period " + str(training_period))
    # print("standard deviations " + str(standard_deviations))
    df = pd.DataFrame(list(results), columns=["Buy and Hold","Strategy","Longs","Sells","Shorts","Covers","Stdev_Strategy","Stdev_Hold","Stock"]) # Create dataframe of results
    df.to_csv("results/Test_Results.csv", index=False) # Save results to csv
    # for stock in list_of_stocks_proccessed:
    #     plot_stocks(stock)