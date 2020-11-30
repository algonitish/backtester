import pandas as pd
import os.path
import queue
import quandl

from abc import ABCMeta, abstractmethod
from event import MarketEvent
from datetime import datetime
from enum import Enum

class DataSource(Enum):
    NASDAQ = "NASDAQ"
    YAHOO = "YAHOO"

class DataHandler(metaclass=ABCMeta):
    """
    DataHandler is an abstract base class providing an interface for
    all subsequent (inherited) data handlers (both live and historic).

    The goal of a (derived) DataHandler object is to output a generated
    set of bars (OLHCVI) for each symbol requested. 

    This will replicate how a live strategy would function as current
    market data would be sent "down the pipe". Thus a historic and live
    system will be treated identically by the rest of the backtesting suite.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N 
        from the latest_symbol list,
        or fewer if less bars are available.
        """
        raise NotImplementedError("Should implement get_latest_bars()")

    @abstractmethod
    def update_bars(self):
        """
        Pushes the latest bar to the latest symbol structure
        for all symbols in the symbol list.
        """
        raise NotImplementedError("Should implement update_bars()")

class HistoricCSVDataHandler(DataHandler):
    """
    HistoricCSVDataHandler is designed to read CSV files for
    each requested symbol from disk and provide an interface
    to obtain the "latest" bar in a manner identical to a live
    trading interface. 
    """
    def __init__(self, events, csv_dir, symbol_list, source=DataSource.NASDAQ):
        self.events = events
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list

        self.symbol_data = {}
        self.symbol_dataframe = {}
        self.latest_symbol_data = {}
        self.all_data = {}
        self.continue_backtest = True

        self.time_col = 1
        self.price_col = 2

        self._open_convert_csv_files(source)

    def _open_convert_csv_files(self, source):
        """
        Opens the CSV files from the data directory, converting
        them into pandas DataFrames within a symbol dictionary.

        For this handler it will be assumed that the data is
        taken from DTN IQFeed. Thus its format will be respected.
        """
        combined_index = None
        for symbol in self.symbol_list:
            if source == DataSource.NASDAQ:
                self.parse_nasdaq_csv(symbol)
            else:
                self.parse_yahoo_csv(symbol)

            if combined_index is None:
                combined_index = self.symbol_data[symbol].index
            else:
                combined_index.union(self.symbol_data[symbol].index)

            self.latest_symbol_data[symbol] = []

        for symbol in self.symbol_list:
            self.symbol_dataframe[symbol] = self.symbol_data[symbol].reindex(index=combined_index, method='pad')
            self.all_data[symbol] = self.symbol_dataframe[symbol].copy()
            self.symbol_data[symbol] = self.symbol_dataframe[symbol].iterrows()

    def _get_new_data(self, symbol):
        """
        Returns the latest bar from the data feed as a tuple of 
        (sybmbol, datetime, open, low, high, close, volume).
        """
        for row in self.symbol_data[symbol]:
            yield tuple([symbol, row[0], row[1][0]])

    def get_latest_data(self, symbol, N=1):
        try:
            return self.latest_symbol_data[symbol][-N:]
        except KeyError:
            print("{symbol} is not a valid symbol.").format(symbol=symbol)

    def update_latest_data(self):
        for symbol in self.symbol_list:
            data = None
            try:
                data = next(self._get_new_data(symbol))
            except StopIteration:
                self.continue_backtest = False
            if data is not None:
                self.latest_symbol_data[symbol].append(data)

        self.events.put(MarketEvent())

    def create_baseline_dataframe(self):
        dataframe = None
        for symbol in self.symbol_list:
            df = self.symbol_dataframe[symbol]
            if dataframe == None:
                dataframe = pd.DataFrame(df['Close'])
                dataframe.columns = [symbol]
            else:
                dataframe[symbol] = pd.DataFrame(df['Close'])
            dataframe[symbol] = dataframe[symbol].pct_change()
            dataframe[symbol] = (1.0 + dataframe[symbol]).cumprod()

        return dataframe

    def parse_yahoo_csv(self, symbol):
        self.symbol_data[symbol] = pd.read_csv(os.path.join(self.csv_dir, symbol + '.csv'), header=0, index_col=0, parse_dates=True)

    def parse_nasdaq_csv(self, symbol):
        tmp = pd.read_csv(os.path.join(self.csv_dir, symbol + '.csv'), header=0, index_col=0, parse_dates=True).iloc[::-1]
        self.symbol_data[symbol] = pd.DataFrame(tmp['Closing price'])
        self.symbol_data[symbol].columns = ['Close']
        # self.symbol_data[symbol]['Open'] = tmp['Closing price']
        # self.symbol_data[symbol]['High'] = tmp['High price']
        # self.symbol_data[symbol]['Low'] = tmp['Low price']
        self.symbol_data[symbol]['Close'] = tmp['Closing price']
        # self.symbol_data[symbol]['Adj Close'] = tmp['Closing price']
        # self.symbol_data[symbol]['Volume'] = tmp['Total volume']
        self.symbol_data[symbol] = self.symbol_data[symbol][self.symbol_data[symbol]['Close'] > 0.0]

class QuandlDataHandler(DataHandler):
    def __init__(self, events, symbol_list, api_key, start_date='2000-01-01', end_date=None):
        quandl.ApiConfig.api_key = api_key
        self.events = events
        self.symbol_list = symbol_list
        self.start_date = start_date
        self.end_date = end_date
        if self.end_date == None:
            self.end_date = datetime.today().strftime('%Y-%m-%d')

        self.symbol_data = {}
        self.symbol_dataframe = {}
        self.latest_symbol_data = {}
        self.all_data = {}
        self.continue_backtest = True

        self.time_col = 1
        self.price_col = 2

        self._load_convert_quandl_data()

    def _load_convert_quandl_data(self):
        combined_index = None
        for symbol in self.symbol_list:
            self._get_nasdaq_data(symbol)

            if combined_index is None:
                combined_index = self.symbol_data[symbol].index
            else:
                combined_index.union(self.symbol_data[symbol].index)

            self.latest_symbol_data[symbol] = []

        for symbol in self.symbol_list:
            self.symbol_dataframe[symbol] = self.symbol_data[symbol].reindex(index=combined_index, method='pad')
            self.all_data[symbol] = self.symbol_dataframe[symbol].copy()
            self.symbol_data[symbol] = self.symbol_dataframe[symbol].iterrows()

    def _get_new_data(self, symbol):
        for row in self.symbol_data[symbol]:
            yield tuple([symbol, row[0], row[1][0]])

    def get_latest_data(self, symbol, N=1):
        try:
            return self.latest_symbol_data[symbol][-N:]
        except KeyError:
            print("{symbol} is not a valid symbol.").format(symbol=symbol)

    def update_latest_data(self):
        for symbol in self.symbol_list:
            data = None
            try:
                data = next(self._get_new_data(symbol))
            except StopIteration:
                self.continue_backtest = False
            if data is not None:
                self.latest_symbol_data[symbol].append(data)

        self.events.put(MarketEvent())

    def create_baseline_dataframe(self):
        dataframe = None
        for symbol in self.symbol_list:
            df = self.symbol_dataframe[symbol]
            if dataframe == None:
                dataframe = pd.DataFrame(df['Close'])
                dataframe.columns = [symbol]
            else:
                dataframe[symbol] = pd.DataFrame(df['Close'])
            dataframe[symbol] = dataframe[symbol].pct_change()
            dataframe[symbol] = (1.0 + dataframe[symbol]).cumprod()

        return dataframe

    def _get_nasdaq_data(self, symbol):
        self.symbol_data[symbol] = quandl.get('NASDAQOMX/' + symbol, start_date=self.start_date, end_date=self.end_date)
        self.symbol_data[symbol].drop(columns=['High', 'Low', 'Total Market Value', 'Dividend Market Value'], inplace=True)
        self.symbol_data[symbol].columns = ['Close']
        self.symbol_data[symbol].index.names = ['Date']
        self.symbol_data[symbol] = self.symbol_data[symbol][self.symbol_data[symbol]['Close'] > 0.0]
