import pandas as pd
import time
import multiprocessing as mp

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
        
        if(lookback['close'][today] < lookback['SMA_25'][today]): # If today's price is above the upper Bollinger Band, enter a short position
            if(lookback['close'][today] > lookback['SMA_250'][today]): # If today's price is below the lower Bollinger Band, exit the position
                if(account.buying_power < 100):
                    for position in account.positions: # Close all current positions
                        account.close_position(position, 1, lookback['close'][today])
                    lookback["sells"][today] = lookback['close'][today]

        elif(lookback['close'][today] > lookback['SMA_250'][today]):
            if(lookback['close'][today-1] <= lookback['SMA_250'][today-1]):
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
        df['TP'] = (df['close'] + df['low'] + df['high'])/3 # Calculate Typical Price
        # df['std'] = df['TP'].rolling(training_period).std() # Calculate Standard Deviation
        # df['MA-TP'] = df['TP'].rolling(training_period).mean() # Calculate Moving Average of Typical Price
        # df['BOLU'] = df['MA-TP'] + standard_deviations*df['std'] # Calculate Upper Bollinger Band
        # df['BOLD'] = df['MA-TP'] - standard_deviations*df['std'] # Calculate Lower Bollinger Band
        
        df['buys'] = "" # Create a column to store the number of buys
        df['sells'] = "" # Create a column to store the number of sells

        # # RSI https://www.roelpeters.be/many-ways-to-calculate-the-rsi-in-python-pandas/
        # close_delta = df['close'].diff()
        # up = close_delta.clip(lower=0)
        # down = -1 * close_delta.clip(upper=0)

        # # Exponential 
        # ma_up = up.ewm(com = training_period - 1, adjust=True, min_periods = training_period).mean()
        # ma_down = down.ewm(com = training_period - 1, adjust=True, min_periods = training_period).mean()
        # rsi = ma_up / ma_down
        # rsi = 100 - (100/(1 + rsi))
        # df["EMA_RSI"] = rsi

        # # SMA
        # ma_up = up.rolling(window = training_period).mean()
        # ma_down = down.rolling(window = training_period).mean()

        # rsi = ma_up / ma_down
        # rsi = 100 - (100/(1 + rsi))
        # df["SMA_RSI"] = rsi

        df['SMA_250'] = df['TP'].rolling(250).mean() # Calculate Moving Average of Typical Price
        df["SMA_25"] = df['TP'].ewm(25).mean() # Calculate Moving Average of Typical Price

        
        
    
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
    list_of_stocks = ["GME_2020-04-19_2022-03-10_1min"] 
    # list_of_stocks = ["AAPL_2020-04-18_2022-03-09_60min"]
    # list_of_stocks = ["TSLA_2020-03-01_2022-01-20_1min", "AAPL_2020-03-24_2022-02-12_1min"] # List of stock data csv's to be tested, located in "data/" folder 
    list_of_stocks_proccessed = preprocess_data(list_of_stocks) # Preprocess the data
    results = tester.test_array(list_of_stocks_proccessed, logic, chart=True) # Run backtest on list of stocks using the logic function
    print("training period " + str(training_period))
    print("standard deviations " + str(standard_deviations))
    df = pd.DataFrame(list(results), columns=["Buy and Hold","Strategy","Longs","Sells","Shorts","Covers","Stdev_Strategy","Stdev_Hold","Stock"]) # Create dataframe of results
    df.to_csv("results/Test_Results.csv", index=False) # Save results to csv
    for stock in list_of_stocks_proccessed:
        plot_stocks(stock)
        