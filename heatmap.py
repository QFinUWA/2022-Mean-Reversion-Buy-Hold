import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()
import pandas as pd
import matplotlib.pyplot as plt

df =pd.read_csv("results/AAPL.csv")
df[["Stock","RSISMA"]] = df['Stock'].str.split('_', 1, expand=True)

df = df[["RSISMA",'Buy and Hold','Strategy']]
aggregation_functions = {'Buy and Hold': 'mean', 'Strategy': 'mean'}
df = df.groupby(df["RSISMA"]).aggregate(aggregation_functions)
df.to_csv("example.csv")
df = df.reset_index()
print(df.head())
df[["RSI","SMA"]] = df['RSISMA'].str.split('-', 1, expand=True)


df["RSI"] =pd.to_numeric(df["RSI"])
df["SMA"] =pd.to_numeric(df["SMA"])
df["Strategy"] =pd.to_numeric(df["Strategy"])
df.to_csv("example.csv")
heat_map = df.pivot(index='RSI', columns="SMA", values='Strategy')
ax = sns.heatmap(heat_map)

plt.show()
