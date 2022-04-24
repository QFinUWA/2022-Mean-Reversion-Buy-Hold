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


SMA_WINDOW = np.arange(0,305,5)
RSI_WINDOW = np.arange(0,505,5)
training_period = max([max(SMA_WINDOW),max(RSI_WINDOW)]) # How far the rolling average takes into calculation

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
    price = lookback['close'][today]
    if(today > training_period): # If the lookback is long enough to calculate the Bollinger Bands
        
        if(price < lookback['SMA'][today]): #if the current price is below the 250 SMA we look for short positions
             #we only want to evaluate potential longs if we have capital to do so
            if(lookback['RSI'][today] > 20 and account.buying_power > 0): #if the RSI is above 20 and as such the stock is not 'oversold' we can look to short
                account.enter_position('short', (account.buying_power), lookback['close'][today]) #enter a short position
        if(lookback['RSI'][today] <= 20): # the rsi of the current price must be below 20 and as such we look to close any shorts we have as the stock is now oversold and highly likely to return to the mean
            for position in account.positions: # Close all current positions
                if(position.type_ == 'short'):
                    account.close_position(position, 1, lookback['close'][today])

                         
        if(price > lookback['SMA'][today]): # means that the current price must be above the 250 SMA and as such we look for long positions
            if(lookback['RSI'][today] <= 80 and account.buying_power > 0): # If the RSI is below 80 the stock is not yet 'overbought' and theres potentially more bullish movement we can take advantage on with a long position
                account.enter_position('long', (account.buying_power), lookback['close'][today])
        
        if(lookback['RSI'][today] <= 20 and price > lookback['SMA'][today]): # the rsi of the current price must be below 20 and as such we look to close any shorts we have as the stock is now oversold and highly likely to return to the mean
            for position in account.positions: # Close all current positions
                if(position.type_ == 'long'):
                    account.close_position(position, 1, lookback['close'][today])
                
        
        
            
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
            difference = (df['close'].diff(1).dropna())
            
            positive_change = 0 * difference
            negative_change = 0 * difference
            
            positive_change[difference > 0] = difference[difference > 0]
            negative_change[difference < 0] = difference[difference < 0]
            
            positive_change_exponential = positive_change.ewm(com=rsi_window, min_periods=rsi_window).mean()
            negative_change_exponential = negative_change.ewm(com=rsi_window, min_periods=rsi_window).mean()
            
            rs = abs(positive_change_exponential / negative_change_exponential)
            
            rsi = 100 - 100/(1 + rs)
            df['RSI'] = rsi
            df['SMA'] = df['TP'].rolling(sma_window).mean()

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
    for rsi_window in RSI_WINDOW:
        for sma_window in SMA_WINDOW:
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
    "AAPL",
    "TSLA"]
    # List of stock data csv's to be tested, located in "data/" folder 
    for stock in list_of_stocks:
        name = f'results/{stock}{min(SMA_WINDOW)}-{max(SMA_WINDOW)}.csv'
        if not os.path.exists(name):
            list_of_stocks_proccessed = preprocess_data(stock) # Preprocess the data
            results = tester.test_array(list_of_stocks_proccessed, logic, chart=False) # Run backtest on list of stocks using the logic function
            df = pd.DataFrame(list(results), columns=["Buy and Hold","Strategy","Longs","Sells","Shorts","Covers","Stdev_Strategy","Stdev_Hold","Stock"]) # Create dataframe of results
            df.to_csv(name, index=False) # Save results to csv
    print(f"timetaken: {time.time()-starttime} seconds")
