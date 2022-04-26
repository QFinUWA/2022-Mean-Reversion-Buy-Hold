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

training_period = 50 # How far the rolling average takes into calculation
standard_deviations = 3.5 # Number of Standard Deviations from the mean the Bollinger Bands sit

'''
logic() function:
    Context: Called for every row in the input data.
    Input:  account - the account object
            lookback - the lookback dataframe, containing all data up until this point in time
    Output: none, but the account object will be modified on each call
'''
def logic(account, lookback): # Logic function to be used for each time interval in backtest 
    
    today = len(lookback)-1
    price = lookback['close'][today]
     
    if(lookback['position'][today] == 1):
        for position in account.positions: # Close all current positions
            account.close_position(position, 1, lookback['close'][today])
        account.enter_position('long', account.buying_power, price)

            
    if(lookback['position'][today] == -1):
        
        for position in account.positions: # Close all current positions
            account.close_position(position, 1, lookback['close'][today])
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
        df['SMAFAST'] = df['close'].ewm(span = (28*24), adjust=False, min_periods=1).mean()
        df['SMASLOW'] = df['close'].ewm(span = (68*24), adjust=False, min_periods=1).mean()
        df['longsignal'] = np.where(df['SMAFAST'] > df['SMASLOW'], 1.0, 0.0)
        df['position'] = df['longsignal'].diff()
        df['positiontype'] = '0'
        df['buys'] = "" # Create a column to store the number of buys
        df['sells'] = "" # Create a column to store the number of sells
        df['covers'] = "" # Create a column to store the number of buys
        df['shorts'] = "" # Create a column to store the number of sells
        
        plt.figure(figsize = (20,10))
        # plot close price, short-term and long-term moving averages 
        df
        df['close'].plot(color = 'k', label = 'Close Price')
        df['SMAFAST'].plot(color = 'b',label = '20-day SMA'); 
        df['SMASLOW'].plot(color = 'g', label = '50-day SMA');
        
        # plot ‘buy' signals
        plt.plot(df[df['position'] == 1].index, 
                df['SMAFAST'][df['position'] == 1], 
                '^', markersize = 15, color = 'g', label = 'buy')
        # plot ‘sell' signals
        plt.plot(df[df['position'] == -1].index, 
                df['SMAFAST'][df['position'] == -1], 
                'v', markersize = 15, color = 'r', label = 'sell')

        plt.ylabel('Price', fontsize = 15 )
        plt.xlabel('Date', fontsize = 15 )
        plt.title(stock, fontsize = 20)
        plt.legend()
        plt.grid()
        plt.show()
        list_of_stocks_processed.append(stock + "_Processed")
        df.to_csv("data/" + stock + "_Processed.csv", index=False) # Save to CSV
        
        
    return list_of_stocks_processed


if __name__ == "__main__":
    # list_of_stocks = ["TSLA_2020-03-01_2022-01-20_1min"] 
    list_of_stocks = ["DIS", "AMZN", "JPM", "GOOG", "PEP", "NVDA"] # List of stock data csv's to be tested, located in "data/" folder 
    list_of_stocks_proccessed = preprocess_data(list_of_stocks) # Preprocess the data
    results = tester.test_array(list_of_stocks_proccessed, logic, chart=True) # Run backtest on list of stocks using the logic function

    print("training period " + str(training_period))
    print("standard deviations " + str(standard_deviations))
    df = pd.DataFrame(list(results), columns=["Buy and Hold","Strategy","Longs","Sells","Shorts","Covers","Stdev_Strategy","Stdev_Hold","Stock"]) # Create dataframe of results
    df.to_csv("results/Test_Results.csv", index=False) # Save results to csv
    # for stock in list_of_stocks_proccessed:
    #     plot_stocks(stock)