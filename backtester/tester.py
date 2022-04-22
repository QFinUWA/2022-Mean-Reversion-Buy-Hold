from turtle import back
import pandas as pd
from backtester import engine
import multiprocessing as mp
import time
from functools import partial
import os

# Function used to backtest each stock
# Parameters: stock - the name of the stock data csv to be tested
#             logic - the logic function to be used
def backtest_stock(results, stock, logic, chart):
    try:
        lock = mp.Lock() # Lock used to prevent errors with multiprocessing
        df = pd.read_csv("data/" + stock + ".csv", parse_dates=[0]) # Read the csv file into a dataframe to be tested
        backtest = engine.backtest(df) # Create a backtest object with the data from the csv
        backtest.start(5000, logic) # Start the backtest with the provided logic function
        lock.acquire()
        data = backtest.results() # Get the results of the backtest
        data.extend([stock]) # Add the stock name to the results for easy comparison
        results.append(data) # Add the results to the list of results
        if chart == True:
            backtest.chart(title=stock + "_results") # Chart the results
        lock.release()
        os.remove("data/" + stock + ".csv")
    except KeyboardInterrupt:
        print("done")

    return data # Return the results

# Function used to test an array of stocks
# Parameters: arr - the array of stock data csv's to be tested
#             logic - the logic function to be used

def test_array(arr, logic, chart):
    manager = mp.Manager() # Create a multiprocessing manager
    results = manager.list() # Create a list to store the results

    args = []
    for stock in arr: # For each stock in the array
        args.append((results, stock, logic, chart))
        
    with mp.Pool(15) as pool:
        pool.starmap(backtest_stock,args)
    
    return results # Return the results
