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

# Kanes
# SMAFAST = np.arange(0,75,5)
# SMASLOW = np.arange(0,155,5)

# Harrys
SMAFAST = np.arange(1, 306, 5)
SMASLOW = np.arange(1,506, 5)

# Best Parameters is SMA FAST at 20 and SMA SLOW at 119 for average of about 90 percent return
training_period = max([max(SMAFAST),max(SMASLOW)]) # How far the rolling average takes into calculation

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
        if(account.buying_power > 0):
            account.enter_position('long', account.buying_power, price)

            
    if(lookback['position'][today] == -1):
        for position in account.positions: # Close all current positions
            account.close_position(position, 1, lookback['close'][today])
        if(account.buying_power > 0):
            account.enter_position('short', account.buying_power, price)
                
        

def create_csvs(stock,sma_slow,sma_fast):
    try:
        if not os.path.exists(f'data/{stock}_{sma_slow}-{SMAFAST}.csv'): 
            df = pd.read_csv("original_data/" + stock + ".csv", parse_dates=[0])
            df = df.iloc[::60, :]
            df['EMAFAST'] = df['close'].ewm(span = (sma_fast*24), adjust=False, min_periods=1).mean()
            df['EMASLOW'] = df['close'].ewm(span = (sma_slow*24), adjust=False, min_periods=1).mean()
            df['longsignal'] = np.where(df['EMAFAST'] > df['EMASLOW'], 1.0, 0.0)
            df['position'] = df['longsignal'].diff()


            df.to_csv(f'data/{stock}_{sma_slow}-{sma_fast}.csv', index=False) # Save to CSV
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
    for sma_slow in SMASLOW:
        for sma_fast in SMAFAST:
            args.append((stock,sma_slow,sma_fast))
            list_of_stocks_processed.append(f'{stock}_{sma_slow}-{sma_fast}')
    with mp.Pool(15) as pool:
        pool.starmap(create_csvs,args)
        
    return list_of_stocks_processed


if __name__ == "__main__":
    
    # print(len(np.arange(0,80,5)))
    # print(len(np.arange(75,155,5)))

    starttime = time.time()
    payload=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    first_table = payload[0]
    df = first_table
    symbols = df['Symbol'].values.tolist()
    common_stocks = symbols[10:]
    list_of_stocks = common_stocks
    # List of stock data csv's to be tested, located in "data/" folder 
    for stock in list_of_stocks:
        name = f'results/{stock}{min(SMASLOW)}-{max(SMASLOW)}.csv'
        if not os.path.exists(name):
            list_of_stocks_proccessed = preprocess_data(stock) # Preprocess the data
            results = tester.test_array(list_of_stocks_proccessed, logic, chart=False) # Run backtest on list of stocks using the logic function
            df = pd.DataFrame(list(results), columns=["Buy and Hold","Strategy","Longs","Sells","Shorts","Covers","Stdev_Strategy","Stdev_Hold","Stock"]) # Create dataframe of results
            df.to_csv(name, index=False) # Save results to csv
    print(f"timetaken: {time.time()-starttime} seconds")