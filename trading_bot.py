# trading_bot.py

import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
import logging

class TradingBot:
    def __init__(self, stock_tickers, benchmark_ticker, interval='1wk'):
        self.stock_tickers = stock_tickers
        self.benchmark_ticker = benchmark_ticker
        self.interval = interval
        self.all_data = {}
        self.log = []
        logging.basicConfig(filename='trading_bot.log', level=logging.INFO)

    def _log_message(self, message):
        """Logs a message to the instance's log."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        full_message = f"[{timestamp}] {message}"
        logging.info(full_message)
        self.log.append(full_message)

    def _fetch_and_prepare_data(self):
        """Fetches market data based on user's stock list and interval."""
        self._log_message(f"Fetching {self.interval} data for {len(self.stock_tickers)} tickers...")
        if not self.stock_tickers:
            self._log_message("Error: Stock list is empty. Cannot fetch data.")
            return False

        all_tickers = self.stock_tickers + [self.benchmark_ticker]
        
        # Adjust data download period based on interval
        days_to_fetch = 730 if self.interval == '1wk' else 200
        start_date = (datetime.now() - timedelta(days=days_to_fetch)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')

        data = yf.download(all_tickers, start=start_date, end=end_date, interval=self.interval, auto_adjust=True, progress=False)

        if data.empty:
            self._log_message("Error: No data downloaded. Check tickers and network connection.")
            return False

        data.columns = data.columns.swaplevel(0, 1)
        benchmark_data = data[self.benchmark_ticker]

        for ticker in self.stock_tickers:
            try:
                stock_data = data[ticker].copy().ffill()
                if stock_data.empty or stock_data.isnull().all().all():
                    continue

                # Calculate indicators
                stock_data.ta.ema(length=11, append=True)
                stock_data.ta.ema(length=21, append=True)
                stock_data.ta.ema(length=50, append=True)
                stock_data.ta.rsi(length=14, append=True)
                stock_data.ta.adx(length=14, append=True)
                stock_data.ta.donchian(lower_length=20, upper_length=20, append=True)
                stock_data['Volume_MA10'] = stock_data['Volume'].rolling(window=10).mean()

                # Calculate Relative Strength
                roll_period = 50 if self.interval == '1wk' else 10 # Shorter lookback for daily
                stock_ret = stock_data['Close'].pct_change().rolling(roll_period).sum()
                bench_ret = benchmark_data['Close'].pct_change().rolling(roll_period).sum()
                stock_data['RS'] = stock_ret / bench_ret

                self.all_data[ticker] = stock_data.dropna()
            except Exception as e:
                self._log_message(f"Warning: Could not process data for {ticker}: {e}")
        
        self._log_message("Data preparation complete.")
        return True

    def get_analysis(self):
        """Executes the data fetching and preparation steps."""
        if not self._fetch_and_prepare_data():
            return None, self.log
        return self.all_data, self.log
