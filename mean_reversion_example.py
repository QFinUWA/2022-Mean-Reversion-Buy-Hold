from calendar import c
from turtle import color
from numpy import diff
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt

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
    yesterday =len(lookback)-2
    price = lookback['close'][today]
    if(today > training_period): # If the lookback is long enough to calculate the Bollinger Bands
        
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
def preprocess_data(list_of_stocks):
    list_of_stocks_processed = []
    for stock in list_of_stocks:
        df = pd.read_csv("original_data/" + stock + ".csv", parse_dates=[0])
        print(df.head)
        
        df['TP'] = (df['close'] + df['low'] + df['high'])/3 # Calculate Typical Price
        df['std'] = df['TP'].rolling(training_period).std() # Calculate Standard Deviation
        df['MA-TP'] = df['TP'].rolling(training_period).mean() # Calculate Moving Average of Typical Price
        df['BOLU'] = df['MA-TP'] + standard_deviations*df['std'] # Calculate Upper Bollinger Band
        df['BOLD'] = df['MA-TP'] - standard_deviations*df['std'] # Calculate Lower Bollinger Band
        difference = (df['close'].diff(1).dropna())
        
        positive_change = 0 * difference
        negative_change = 0 * difference
        
        positive_change[difference > 0] = difference[difference > 0]
        negative_change[difference < 0] = difference[difference < 0]
        
        positive_change_exponential = positive_change.ewm(com=training_period-1, min_periods=training_period).mean()
        negative_change_exponential = negative_change.ewm(com=training_period-1, min_periods=training_period).mean()
        
        rs = abs(positive_change_exponential / negative_change_exponential)
        
        rsi = 100 - 100/(1 + rs)
        df['RSI'] = rsi
        
        df['buys'] = "" # Create a column to store the number of buys
        df['sells'] = "" # Create a column to store the number of sells
        
        df['SMAFAST'] = df['TP'].rolling((50*26)*4).mean() # Calculate Moving Average of Typical Price
        df["SMASLOW"] = df['TP'].rolling((250*26)*4).mean() # Calculate Moving Average of Typical Price
        
        list_of_stocks_processed.append(stock + "_Processed")
        df.to_csv("data/" + stock + "_Processed.csv", index=False) # Save to CSV
        
        
    return list_of_stocks_processed

def plot_stocks(df):
    df = pd.read_csv("data/" + stock +'.csv', parse_dates=[0])
    plt.title('Price chart ')
    plt.plot(df['date'], df['SMASLOW'], color='blue')
    plt.plot(df['date'], df['SMAFAST'], color='green')
    plt.scatter(df['date'], df['buys'],c="red")
    plt.scatter(df["date"], df["sells"],c="purple")
    plt.plot(df['date'], df['close'], color='black')
    plt.show()

if __name__ == "__main__":
    # list_of_stocks = ["TSLA_2020-03-01_2022-01-20_1min"] 
    list_of_stocks = ["DIS", "AMZN", "JPM", "GOOG", "PEP", "NVDA"] # List of stock data csv's to be tested, located in "data/" folder 
    list_of_stocks_proccessed = preprocess_data(list_of_stocks) # Preprocess the data
    results = tester.test_array(list_of_stocks_proccessed, logic, chart=True) # Run backtest on list of stocks using the logic function

    print("training period " + str(training_period))
    print("standard deviations " + str(standard_deviations))
    df = pd.DataFrame(list(results), columns=["Buy and Hold","Strategy","Longs","Sells","Shorts","Covers","Stdev_Strategy","Stdev_Hold","Stock"]) # Create dataframe of results
    df.to_csv("results/Test_Results.csv", index=False) # Save results to csv
    for stock in list_of_stocks_proccessed:
        plot_stocks(stock)