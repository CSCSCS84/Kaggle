from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import chart_studio.plotly as py
import cufflinks as cf

start = datetime.datetime(2006, 1, 1)
end = datetime.datetime(2016, 1, 1)

#all_banks = pd.read_csv("data//all_banks.csv",encoding='utf-8')
#print(all_banks.head())
#Bank of America
BAC = data.DataReader("BAC", 'yahoo', start, end)
print(BAC.describe())
print(BAC.info())
print(BAC.head())
# CitiGroup
C = data.DataReader("C", 'yahoo', start, end)
print(C.head())
# Goldman Sachs
GS = data.DataReader("GS", 'yahoo', start, end)
# JPMorgan Chase
JPM = data.DataReader("JPM", 'yahoo', start, end)
# Morgan Stanley
MS = data.DataReader("MS", 'yahoo', start, end)
# Wells Fargo
WFC = data.DataReader("WFC", 'yahoo', start, end)

tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']

bank_stocks=pd.concat([BAC, C, GS, JPM, MS, WFC], axis=1, keys=tickers)
bank_stocks.columns.names = ['Bank Ticker','Stock Info']
print(bank_stocks.head())
print(bank_stocks.index.value_counts())
max_values=bank_stocks.xs(key="Close", axis=1, level="Stock Info").max()
print(max_values)

returns=pd.DataFrame(index=bank_stocks.index)
print(returns.head())

for tick in tickers:
    columns_name= tick + " Return"
    returns.loc[:,columns_name]= bank_stocks.loc[:,tick]['Close'].pct_change()


print(returns.head())
#sns.pairplot(data=returns[1:])
#plt.show()
print()
print(returns.idxmax(axis=0))
print(returns.idxmin(axis=0))
print(returns.std(axis=0))

#sns.displot(data=returns.loc["2015-01-01":"2015-12-31","MS Return"],bins=100,kde=True)
#sns.displot(data=returns.loc["2008-01-01":"2008-12-31","C Return"],bins=100,kde=True)
closeprices=bank_stocks.xs(key="Close", axis=1, level="Stock Info")
print(closeprices.head())
#sns.lineplot(data=closeprices,lw=1)

#moving 30 days average
for tick in tickers:
    columns_name= tick + " Moving Avg 30 days"
    returns.loc[:,columns_name]= bank_stocks.loc[:,tick]['Close'].rolling(window=30).mean()
print(returns.head(50))
#sns.lineplot(data=returns.loc["2008-01-01":"2008-12-31","BAC Moving Avg 30 days"])
#sns.lineplot(data=BAC.loc["2008-01-01":"2008-12-31","Close"])

#Create a heatmap of the correlation between the stocks Close Price.
#sns.heatmap(data=closeprices.corr(),annot=True)
#sns.clustermap(data=closeprices.corr(),annot=True)


#create a candle plot of Bank of America's stock from Jan 1st 2015 to Jan 1st 2016
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
BAC.loc['2015-01-01':'2016-01-01',['Open', 'High', 'Low', 'Close']].iplot(kind='candle')
plt.show()