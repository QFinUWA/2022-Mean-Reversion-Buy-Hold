from numpy import diff
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt

# local imports
from backtester import engine, tester
from backtester import API_Interface as api

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
        
        if(lookback['close'][today] < lookback['SMA_250'][today]): #if the current price is below the 250 SMA we look for short positions
             #we only want to evaluate potential longs if we have capital to do so
            if(lookback['RSI'][today] > 20 and account.buying_power > 0): #if the RSI is above 20 and as such the stock is not 'oversold' we can look to short
                account.enter_position('short', (account.buying_power), lookback['close'][today]) #enter a short position
                lookback['shorts'][today] = lookback['close'][today] #set the short row for this price to the current position and price for graphing
            elif(lookback['RSI'][today] <= 20): # the rsi of the current price must be below 20 and as such we look to close any shorts we have as the stock is now oversold and highly likely to return to the mean
                for position in account.positions: # Close all current positions
                        account.close_position(position, 1, lookback['close'][today])
                lookback['covers'][today] = lookback['close'][today] #set the covers row for this price to the current position and price for graphing
                         
        if(lookback['close'][today] > lookback['SMA_250'][today]): #Else means that the current price must be above the 250 SMA and as such we look for long positions
            if(lookback['RSI'][today] <= 80 and account.buying_power > 0): # If the RSI is below 80 the stock is not yet 'overbought' and theres potentially more bullish movement we can take advantage on with a long position
                account.enter_position('long', (account.buying_power), lookback['close'][today])
                lookback['buys'][today] = lookback['close'][today] #set the long row for this price to the current position and price for graphing
            elif(lookback['RSI'][today] >= 80):
                for position in account.positions: # Close all current positions
                        account.close_position(position, 1, lookback['close'][today])
                lookback['sells'][today] = lookback['close'][today] #set the covers row for this price to the current position and price for graphing
                


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
        df['std'] = df['TP'].rolling(training_period).std() # Calculate Standard Deviation
        df['MA-TP'] = df['TP'].rolling(training_period).mean() # Calculate Moving Average of Typical Price
        df['BOLU'] = df['MA-TP'] + standard_deviations*df['std'] # Calculate Upper Bollinger Band
        df['BOLD'] = df['MA-TP'] - standard_deviations*df['std'] # Calculate Lower Bollinger Band
        difference = (df['close'].diff(1).dropna())
        
        positive_change = 0 * difference
        negative_change = 0 * difference
        
        positive_change[difference > 0] = difference[difference > 0]
        negative_change[difference < 0] = difference[difference < 0]
        
        positive_change_exponential = positive_change.ewm(com=14, min_periods=training_period).mean()
        negative_change_exponential = negative_change.ewm(com=14, min_periods=training_period).mean()
        
        rs = abs(positive_change_exponential / negative_change_exponential)
        
        rsi = 100 - 100/(1 + rs)
        df['RSI'] = rsi
        
        df['buys'] = "" # Create a column to store the number of buys
        df['sells'] = "" # Create a column to store the number of sells
        df['shorts'] = "" # Create a column to store the number of shorts
        df['covers'] = "" # Create a column to store the number of covers
        
        df['SMA_250'] = df['TP'].rolling(250).mean() # Calculate Moving Average of Typical Price
        df["SMA_25"] = df['TP'].ewm(25).mean() # Calculate Moving Average of Typical Price
        
        list_of_stocks_processed.append(stock + "_Processed")
        df.to_csv("data/" + stock + "_Processed.csv", index=False) # Save to CSV
        
        
    return list_of_stocks_processed

def plot_stocks(df):
    df = pd.read_csv("data/" + stock +'.csv', parse_dates=[0])
    plt.title('Price chart ')
    plt.plot(df['date'], df['SMA_250'])
    plt.plot(df['date'], df['RSI'])
    # plt.plot(df['date'], df['SMA_25'])
    plt.scatter(df['date'], df['buys'],c="red")
    plt.scatter(df["date"], df['sells'],c="purple")
    plt.scatter(df['date'], df['shorts'],c="yellow")
    plt.scatter(df["date"], df['covers'],c="black")
    plt.plot(df['date'], df['close'])
    plt.show()

if __name__ == "__main__":
    list_of_stocks = ["TSLA_2020-03-01_2022-01-20_1min", "GME_2020-04-30_2022-03-21_1min"]
     
    # list_of_stocks = ["GME_2020-04-19_2022-03-10_1min"] # List of stock data csv's to be tested, located in "data/" folder 
    list_of_stocks_proccessed = preprocess_data(list_of_stocks) # Preprocess the data
    results = tester.test_array(list_of_stocks_proccessed, logic, chart=True) # Run backtest on list of stocks using the logic function

    print("training period " + str(training_period))
    print("standard deviations " + str(standard_deviations))
    df = pd.DataFrame(list(results), columns=["Buy and Hold","Strategy","Longs","Sells","Shorts","Covers","Stdev_Strategy","Stdev_Hold","Stock"]) # Create dataframe of results
    df.to_csv("results/Test_Results.csv", index=False) # Save results to csv
    for stock in list_of_stocks_proccessed:
        plot_stocks(stock)