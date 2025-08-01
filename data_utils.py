# data_utils.py

import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
import logging
import numpy as np

logger = logging.getLogger(__name__)

def get_data_with_indicators(tickers: list, benchmark_ticker: str, interval: str):
    """
    Centralized function to fetch market data and calculate all necessary technical indicators.
    
    Args:
        tickers (list): A list of stock tickers to process.
        benchmark_ticker (str): The ticker for the market benchmark (e.g., '^NSEI').
        interval (str): The data interval ('1d', '1wk', etc.).

    Returns:
        dict: A dictionary where keys are tickers and values are DataFrames with calculated indicators.
              Returns None if the initial data fetch fails.
    """
    if not tickers:
        logger.warning("get_data_with_indicators called with an empty ticker list.")
        return {}

    all_tickers_to_fetch = list(set(tickers + [benchmark_ticker]))
    
    # Determine history based on interval
    days_to_fetch = 730 if interval == '1wk' else 365
    start_date = (datetime.now() - timedelta(days=days_to_fetch)).strftime('%Y-%m-%d')
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d') # Fetch up to today

    logger.info(f"Fetching {interval} data for {len(all_tickers_to_fetch)} tickers from {start_date} to {end_date}")

    try:
        data = yf.download(all_tickers_to_fetch, start=start_date, end=end_date, interval=interval, auto_adjust=True, progress=False)
        if data.empty:
            logger.error("yf.download returned an empty DataFrame.")
            return None
    except Exception as e:
        logger.error(f"Failed to download data from yfinance: {e}", exc_info=True)
        return None

    # Standardize column access
    data.columns = data.columns.swaplevel(0, 1)
    
    benchmark_data = data.get(benchmark_ticker)
    if benchmark_data is None or benchmark_data.empty:
        logger.error(f"Benchmark data for {benchmark_ticker} could not be processed.")
        return None

    processed_data = {}
    for ticker in tickers:
        if ticker not in data:
            logger.warning(f"No data for ticker {ticker} in downloaded batch.")
            continue
            
        stock_data = data[ticker].copy().ffill()
        if stock_data.empty or stock_data.isnull().all().all():
            continue

        try:
            # --- Calculate all required indicators ---
            stock_data.ta.ema(length=11, append=True)
            stock_data.ta.ema(length=21, append=True)
            # FIX: Add the missing EMA_40 calculation
            stock_data.ta.ema(length=40, append=True)
            stock_data.ta.ema(length=50, append=True)
            stock_data.ta.rsi(length=14, append=True)
            stock_data.ta.adx(length=14, append=True)
            stock_data.ta.donchian(lower_length=20, upper_length=20, append=True)
            stock_data['Volume_MA10'] = stock_data['Volume'].rolling(window=10).mean()

            # --- Robust Relative Strength (RS) Calculation ---
            roll_period = 50 if interval == '1wk' else 21
            stock_ret = stock_data['Close'].pct_change().rolling(roll_period).sum()
            bench_ret = benchmark_data['Close'].pct_change().rolling(roll_period).sum()
            
            denominator = 1 + bench_ret
            # Use .replace(0, np.nan) to prevent division by zero errors
            stock_data['RS'] = (1 + stock_ret) / denominator.replace(0, np.nan)

            processed_data[ticker] = stock_data.dropna()
        except Exception as e:
            logger.error(f"Could not calculate indicators for {ticker}: {e}", exc_info=True)
            
    return processed_data
