import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

# path = r'results' # use your path
# all_files = glob.glob(os.path.join(path , "/*.csv"))

li = []

for filename in os.listdir('results'):
    df = pd.read_csv(f'results/{filename}', index_col=None, header=0)
    li.append(df)

df = pd.concat(li, axis=0, ignore_index=True)
df.to_csv("finished.csv")
# quit()

# df =pd.read_csv("results/AAPL.csv")
df[["Stock","SLOWFAST"]] = df['Stock'].str.split('_', 1, expand=True)

df = df[["SLOWFAST",'Buy and Hold','Strategy']]
aggregation_functions = {'Buy and Hold': 'mean', 'Strategy': 'mean'}
df = df.groupby(df["SLOWFAST"]).aggregate(aggregation_functions)
df.to_csv("example.csv")
df = df.reset_index()
print(df.max())
df[["SLOW","FAST"]] = df['SLOWFAST'].str.split('-', 1, expand=True)


df["SLOW"] =pd.to_numeric(df["SLOW"])
df["FAST"] =pd.to_numeric(df["FAST"])
df["Strategy"] =pd.to_numeric(df["Strategy"])
df.to_csv("example.csv")
heat_map = df.pivot(index='SLOW', columns="FAST", values='Strategy')
ax = sns.heatmap(heat_map)

plt.show()
