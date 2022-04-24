from re import L
from numpy import diff
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import os
import time
# local imports
from backtester import engine, tester
from backtester import API_Interface as api
from backtester.account import LongPosition


SMAFAST_WINDOW = np.arange(0,30,1)
SMASLOW_WINDOW = np.arange(0,50,1)
training_period = max([max(SMAFAST_WINDOW),max(SMASLOW_WINDOW)]) # How far the rolling average takes into calculation

'''
logic() function:
    Context: Called for every row in the input data.

    Input:  account - the account object
            lookback - the lookback dataframe, containing all data up until this point in time

    Output: none, but the account object will be modified on each call
'''

def logic(account, lookback): # Logic function to be used for each time interval in backtest 
    
    today = len(lookback)-1
    yesterday =len(lookback)-2
    price = lookback['close'][today] # If the lookback is long enough to calculate the Bollinger Bands
        
    if(lookback['SMASLOW'][today] < lookback['SMAFAST'][today] and lookback['SMASLOW'][yesterday] > lookback['SMAFAST'][yesterday]): #If there is a crossover of fast from below the slow to above the slow
        for position in account.positions: # Close all current positions
            account.close_position(position, 1, lookback['close'][today])  
    if(lookback['SMASLOW'][today] > lookback['SMAFAST'][today] and lookback['SMASLOW'][yesterday] < lookback['SMAFAST'][yesterday]): # the rsi of the current price must be below 20 and as such we look to close any shorts we have as the stock is now oversold and highly likely to return to the mean
        if(account.buying_power > 2):
            account.enter_position('long', (account.buying_power), price)
       
                
        
        
            
'''
preprocess_data() function:
    Context: Called once at the beginning of the backtest. TOTALLY OPTIONAL. 
             Each of these can be calculated at each time interval, however this is likely slower.

    Input:  list_of_stocks - a list of stock data csvs to be processed

    Output: list_of_stocks_processed - a list of processed stock data csvs
'''

def create_csvs(stock,rsi_window,sma_window):
    try:
        if not os.path.exists(f'data/{stock}_{rsi_window}-{sma_window}.csv'): 
            df = pd.read_csv("original_data/" + stock + ".csv", parse_dates=[0])
            df = df.iloc[::60, :]
            
            df['TP'] = (df['close'] + df['low'] + df['high'])/3 # Calculate Typical Price
            # difference = (df['close'].diff(1).dropna())
            
            # positive_change = 0 * difference
            # negative_change = 0 * difference
            
            # positive_change[difference > 0] = difference[difference > 0]
            # negative_change[difference < 0] = difference[difference < 0]
            
            # positive_change_exponential = positive_change.ewm(com=rsi_window, min_periods=rsi_window).mean()
            # negative_change_exponential = negative_change.ewm(com=rsi_window, min_periods=rsi_window).mean()
            
            # rs = abs(positive_change_exponential / negative_change_exponential)
            
            # rsi = 100 - 100/(1 + rs)
            # df['RSI'] = rsi
            df['SMAFAST'] = df['close'].rolling((sma_window*24)*2).mean()
            df['SMASLOW'] = df['close'].rolling((rsi_window*24)*2).mean()
            df['buysignal'] = np.where((df['SMAFAST'] > df['SMASLOW']), 1.0, 0.0)
            df['sellsignal'] = np.where((df['SMAFAST'] < df['SMASLOW']), 1.0, 0.0)

            df['position'] = df['buysignal'].diff()
            
            df.to_csv(f'data/{stock}_{rsi_window}-{sma_window}.csv', index=False) # Save to CSV
            
    except KeyboardInterrupt:
        print("done")   
        
         
'''
preprocess_data() function:
    Context: Called once at the beginning of the backtest. TOTALLY OPTIONAL. 
             Each of these can be calculated at each time interval, however this is likely slower.
    Input:  list_of_stocks - a list of stock data csvs to be processed
    Output: list_of_stocks_processed - a list of processed stock data csvs
'''

def preprocess_data(stock):
    list_of_stocks_processed = []
    args = []
    for rsi_window in SMASLOW_WINDOW:
        for sma_window in SMAFAST_WINDOW:
            args.append((stock,rsi_window,sma_window))
            list_of_stocks_processed.append(f'{stock}_{rsi_window}-{sma_window}')
    with mp.Pool(15) as pool:
        pool.starmap(create_csvs,args)
        
    return list_of_stocks_processed


if __name__ == "__main__":
    
    # print(len(np.arange(0,80,5)))
    # print(len(np.arange(75,155,5)))

    starttime = time.time()
    list_of_stocks = [
    "AMZN",
    "AAPL",
    "JNJ",
    "JPM",
    "UNH",
    "V",
    "TSLA"]
    # List of stock data csv's to be tested, located in "data/" folder 
    for stock in list_of_stocks:
        name = f'results/{stock}{min(SMAFAST_WINDOW)}-{max(SMAFAST_WINDOW)}.csv'
        if not os.path.exists(name):
            list_of_stocks_proccessed = preprocess_data(stock) # Preprocess the data
            results = tester.test_array(list_of_stocks_proccessed, logic, chart=False) # Run backtest on list of stocks using the logic function
            df = pd.DataFrame(list(results), columns=["Buy and Hold","Strategy","Longs","Sells","Shorts","Covers","Stdev_Strategy","Stdev_Hold","Stock"]) # Create dataframe of results
            df.to_csv(name, index=False) # Save results to csv
    print(f"timetaken: {time.time()-starttime} seconds")
