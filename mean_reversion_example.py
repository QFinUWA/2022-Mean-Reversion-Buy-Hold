import pandas as pd
import time
import multiprocessing as mp
import numpy as np

# local imports
from backtester import engine, tester
from backtester import API_Interface as api

import matplotlib.pyplot as plt

training_period = 20 # How far the rolling average takes into calculation
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
    
    if(today > training_period): # If the lookback is long enough to calculate the Bollinger Bands
        
        if(account.buying_power < 100):
            if(lookback["+DM"][today]>lookback["-DM"][today] and lookback["SMA_RSI"][today] >= 75):
                for position in account.positions: # Close all current positions
                    account.close_position(position, 1, lookback['close'][today])
                lookback["sells"][today] = lookback['close'][today]

        elif(lookback["SMA_RSI"][today] <= 25):
            if(account.buying_power > 100):
                account.enter_position('long', account.buying_power, lookback['close'][today]) # Enter a long position 
                lookback["buys"][today] = lookback['close'][today]

    
    
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
        df = pd.read_csv("data/" + stock + ".csv", parse_dates=[0])
        df = df.iloc[::60, :]
       

        df['buys'] = "" # Create a column to store the number of buys
        df['sells'] = "" # Create a column to store the number of sells

        # RSI https://www.roelpeters.be/many-ways-to-calculate-the-rsi-in-python-pandas/
        df['TP'] = (df['close'] + df['low'] + df['high'])/3 # Calculate Typical Price
        close_delta = df['close'].diff()
        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)

        # Exponential 
        ma_up = up.ewm(com = training_period - 1, adjust=True, min_periods = training_period).mean()
        ma_down = down.ewm(com = training_period - 1, adjust=True, min_periods = training_period).mean()
        rsi = ma_up / ma_down
        rsi = 100 - (100/(1 + rsi))
        df["EMA_RSI"] = rsi

        # SMA
        ma_up = up.rolling(window = training_period).mean()
        ma_down = down.rolling(window = training_period).mean()
        rsi = ma_up / ma_down
        rsi = 100 - (100/(1 + rsi))
        df["SMA_RSI"] = rsi

        # Directional Movement Index
        # https://python.plainenglish.io/trading-using-python-average-directional-index-adx-aeab999cffe7
        interval = 14
        df['-DM'] = df['low'].shift(1) - df['low']
        df['+DM'] = df['high'] - df['high'].shift(1)
        df['+DM'] = np.where((df['+DM'] > df['-DM']) & (df['+DM']>0), df['+DM'], 0.0)
        df['-DM'] = np.where((df['-DM'] > df['+DM']) & (df['-DM']>0), df['-DM'], 0.0)
        df['TR_TMP1'] = df['high'] - df['low']
        df['TR_TMP2'] = np.abs(df['high'] - df['close'].shift(1))
        df['TR_TMP3'] = np.abs(df['low'] - df['close'].shift(1))
        df['TR'] = df[['TR_TMP1', 'TR_TMP2', 'TR_TMP3']].max(axis=1)
        df['TR'+str(interval)] = df['TR'].rolling(interval).sum()
        df['+DMI'+str(interval)] = df['+DM'].rolling(interval).sum()
        df['-DMI'+str(interval)] = df['-DM'].rolling(interval).sum()
        df['+DI'+str(interval)] = df['+DMI'+str(interval)] /   df['TR'+str(interval)]*100
        df['-DI'+str(interval)] = df['-DMI'+str(interval)] / df['TR'+str(interval)]*100
        df['DI'+str(interval)+'-'] = abs(df['+DI'+str(interval)] - df['-DI'+str(interval)])
        df['DI'+str(interval)] = df['+DI'+str(interval)] + df['-DI'+str(interval)]
        df['DX'] = (df['DI'+str(interval)+'-'] / df['DI'+str(interval)])*100
        df['ADX'+str(interval)] = df['DX'].rolling(interval).mean()
        df['ADX'+str(interval)] =   df['ADX'+str(interval)].fillna(df['ADX'+str(interval)].mean())
        
        del df['TR_TMP1'], df['TR_TMP2'], df['TR_TMP3'], df['TR'], df['TR'+str(interval)]
        del df['+DMI'+str(interval)], df['DI'+str(interval)+'-']
        del df['DI'+str(interval)], df['-DMI'+str(interval)]
        del df['+DI'+str(interval)], df['-DI'+str(interval)]
        del df['DX']
        del df["TP"]



        
        
    
        df.to_csv("data/" + stock + "_Processed.csv", index=False) # Save to CSV
        list_of_stocks_processed.append(stock + "_Processed")
    return list_of_stocks_processed



def plot_stocks(df):
    df = pd.read_csv("data/" + stock +'.csv', parse_dates=[0])
    plt.plot(df['date'], df['close'])
    plt.title('Price chart ')
    plt.plot(df['date'], df['SMA_250'])
    plt.plot(df['date'], df['SMA_25'])
    plt.scatter(df['date'], df['buys'],c="red")
    plt.scatter(df["date"], df["sells"],c="purple")
    plt.show()

if __name__ == "__main__":
    list_of_stocks = ["AAPL_2020-03-24_2022-02-12_1min"] 
    # list_of_stocks = ["AAPL_2020-04-18_2022-03-09_60min"]
    # list_of_stocks = ["TSLA_2020-03-01_2022-01-20_1min", "AAPL_2020-03-24_2022-02-12_1min"] # List of stock data csv's to be tested, located in "data/" folder 
    list_of_stocks_proccessed = preprocess_data(list_of_stocks) # Preprocess the data
    results = tester.test_array(list_of_stocks_proccessed, logic, chart=True) # Run backtest on list of stocks using the logic function
    print("training period " + str(training_period))
    print("standard deviations " + str(standard_deviations))
    df = pd.DataFrame(list(results), columns=["Buy and Hold","Strategy","Longs","Sells","Shorts","Covers","Stdev_Strategy","Stdev_Hold","Stock"]) # Create dataframe of results
    df.to_csv("results/Test_Results.csv", index=False) # Save results to csv
    # for stock in list_of_stocks_proccessed:
    #     plot_stocks(stock)
        