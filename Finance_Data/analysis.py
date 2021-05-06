from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

start = datetime.datetime(2006, 1, 1)
end = datetime.datetime(2016, 1, 1)
global BAC, C, GS, JPM, MS, WFC, bank_stocks, tickers, returns, closeprices


def read_bank_data():
    global BAC, C, GS, JPM, MS, WFC
    # Bank of America
    BAC = data.DataReader("BAC", 'yahoo', start, end)
    # CitiGroup
    C = data.DataReader("C", 'yahoo', start, end)
    # Goldman Sachs
    GS = data.DataReader("GS", 'yahoo', start, end)
    # JPMorgan Chase
    JPM = data.DataReader("JPM", 'yahoo', start, end)
    # Morgan Stanley
    MS = data.DataReader("MS", 'yahoo', start, end)
    # Wells Fargo
    WFC = data.DataReader("WFC", 'yahoo', start, end)


def create_tickers():
    global tickers
    tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']


def create_bank_stocks():
    global bank_stocks
    bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC], axis=1, keys=tickers)
    bank_stocks.columns.names = ['Bank Ticker', 'Stock Info']


def print_max_close_values():
    max_values = bank_stocks.xs(key="Close", axis=1, level="Stock Info").max()
    print(max_values)


def create_returns():
    global returns, tick, columns_name
    returns = pd.DataFrame(index=bank_stocks.index)
    print(returns.head())
    for tick in tickers:
        columns_name = tick + " Return"
        returns.loc[:, columns_name] = bank_stocks.loc[:, tick]['Close'].pct_change()


def print_returns_statistics():
    print(returns.idxmax(axis=0))
    print(returns.idxmin(axis=0))
    print(returns.std(axis=0))


def pairplot_returns():
    sns.pairplot(data=returns[1:])
    plt.show()


def distplot_returns():
    sns.displot(data=returns.loc["2015-01-01":"2015-12-31", "MS Return"], bins=100, kde=True)
    sns.displot(data=returns.loc["2008-01-01":"2008-12-31", "C Return"], bins=100, kde=True)
    plt.show()


def create_closeprices():
    global closeprices
    closeprices = bank_stocks.xs(key="Close", axis=1, level="Stock Info")


def plot_moving_averages():
    global tick, columns_name
    # moving 30 days average
    for tick in tickers:
        columns_name = tick + " Moving Avg 30 days"
        returns.loc[:, columns_name] = bank_stocks.loc[:, tick]['Close'].rolling(window=30).mean()
    print(returns.head(50))
    sns.lineplot(data=returns.loc["2008-01-01":"2008-12-31","BAC Moving Avg 30 days"])
    sns.lineplot(data=BAC.loc["2008-01-01":"2008-12-31","Close"])
    plt.show()

# Create a heatmap and clustermap of the correlation between the stocks Close Price
def plot_heatmap_closeprices():
    sns.heatmap(data=closeprices.corr(), annot=True)
    sns.clustermap(data=closeprices.corr(), annot=True)
    plt.show()


def plot_closeprices():
    sns.lineplot(data=closeprices, lw=1)
    plt.show()


read_bank_data()
create_tickers()
create_bank_stocks()
print(bank_stocks.head())
print(bank_stocks.index.value_counts())

print_max_close_values()
create_returns()
pairplot_returns()
print_returns_statistics()
distplot_returns()
create_closeprices()
plot_closeprices()
plot_moving_averages()
plot_heatmap_closeprices()
