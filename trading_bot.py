# trading_bot.py

import logging
from datetime import datetime
from data_utils import get_data_with_indicators # Use the centralized data utility

class TradingBot:
    def __init__(self, stock_tickers, benchmark_ticker, interval='1wk'):
        self.stock_tickers = stock_tickers
        self.benchmark_ticker = benchmark_ticker
        self.interval = interval
        self.log = []
        logging.basicConfig(filename='trading_bot.log', level=logging.INFO)

    def _log_message(self, message):
        """Logs a message to the instance's log."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        full_message = f"[{timestamp}] {message}"
        logging.info(full_message)
        self.log.append(full_message)

    def get_analysis(self):
        """
        Executes the data fetching and preparation steps using the centralized utility.
        """
        self._log_message("Fetching data and calculating indicators...")
        
        # Call the centralized function to get data with all indicators
        all_data = get_data_with_indicators(
            tickers=self.stock_tickers,
            benchmark_ticker=self.benchmark_ticker,
            interval=self.interval
        )

        if all_data is None:
            self._log_message("Error: Failed to fetch or process market data.")
            return None, self.log
            
        self._log_message("Data preparation complete.")
        return all_data, self.log
