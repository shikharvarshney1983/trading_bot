# stock_screener.py
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from datetime import date, timedelta, datetime
import sqlite3
import numpy as np
import os
import logging
import pytz
import calendar
import holidays

DB_PATH = os.path.join('instance', 'trading.db')
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TIME_CONFIGS = {
    'daily': {'interval': '1d', 'history_days': 365 * 2, 'rpi_period': 50, 'year_lookback': 252, 'quarter_lookback': 63},
    'weekly': {'interval': '1wk', 'history_days': 365 * 5, 'rpi_period': 20, 'year_lookback': 52, 'quarter_lookback': 13},
    'monthly': {'interval': '1mo', 'history_days': 365 * 10, 'rpi_period': 12, 'year_lookback': 12, 'quarter_lookback': 3}
}

def get_last_valid_trading_day(start_date):
    nse_holidays = holidays.country_holidays('IN', subdiv='KA')
    current_date = start_date
    while True:
        if current_date.weekday() < 5 and current_date not in nse_holidays:
            return current_date
        current_date -= timedelta(days=1)

def get_screener_end_date(frequency):
    tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(tz)
    today = now.date()
    market_close_time = datetime(2000, 1, 1, 15, 30).time()
    ideal_end_date = today
    if frequency == 'daily':
        if not (now.weekday() < 5 and now.time() >= market_close_time):
            ideal_end_date = today - timedelta(days=1)
    elif frequency == 'weekly':
        if now.weekday() < 4 or (now.weekday() == 4 and now.time() < market_close_time):
            ideal_end_date = today - timedelta(days=today.weekday() + 3)
    elif frequency == 'monthly':
        _, last_day_num = calendar.monthrange(today.year, today.month)
        is_last_day_of_month = (today.day == last_day_num)
        if not (is_last_day_of_month and now.time() >= market_close_time):
            ideal_end_date = today.replace(day=1) - timedelta(days=1)
    return get_last_valid_trading_day(ideal_end_date)

def get_nse_stocks_from_db():
    db = sqlite3.connect(DB_PATH)
    try:
        symbols_df = pd.read_sql_query("SELECT symbol FROM master_stocks", db)
        return symbols_df['symbol'].tolist()
    finally:
        db.close()

def download_data_in_chunks(tickers, start, end, interval, chunk_size=100):
    """Downloads data in chunks to make it more robust."""
    all_data = pd.DataFrame()
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            data_chunk = yf.download(chunk, start=start, end=end, interval=interval, progress=False, auto_adjust=True)
            if not data_chunk.empty:
                all_data = pd.concat([all_data, data_chunk], axis=1)
        except Exception as e:
            logger.error(f"Failed to download chunk {i//chunk_size + 1}: {e}")
    return all_data

def run_screener_process(frequency='weekly'):
    logger.info(f"Starting {frequency} stock screener process...")
    try:
        config = TIME_CONFIGS[frequency]
        interval, history_days, rpi_period, year_lookback, quarter_lookback = config.values()
        
        all_symbols = get_nse_stocks_from_db()
        if not all_symbols:
            logger.info("No stocks in master list. Aborting.")
            return

        end_date_inclusive = get_screener_end_date(frequency)
        start_date = end_date_inclusive - timedelta(days=history_days)
        end_date_exclusive = end_date_inclusive + timedelta(days=1)
        
        logger.info(f"Running screener for {frequency} with data ending on {end_date_inclusive.strftime('%Y-%m-%d')}")
        
        all_data = download_data_in_chunks(all_symbols + ['^NSEI'], start_date, end_date_exclusive, interval)
        if all_data.empty:
            logger.error("Failed to download any stock data. Aborting screener.")
            return
        
        all_data.columns = pd.MultiIndex.from_tuples([(col[1] if isinstance(col, tuple) else col, col[0] if isinstance(col, tuple) else '') for col in all_data.columns])
        all_data = all_data.reorder_levels([1, 0], axis=1).sort_index(axis=1)

        nifty_data = all_data['^NSEI']['Close']
        all_stocks_data = []

        for symbol in all_symbols:
            if symbol not in all_data: continue
            data = all_data[symbol].copy()
            if data.empty or data['Close'].isnull().all() or len(data) < 50: continue

            data.ta.adx(length=14, append=True)
            data.ta.rsi(length=14, append=True)
            data.ta.ema(length=11, append=True)
            data.ta.ema(length=21, append=True)
            data.ta.ema(length=40, append=True)
            data['Volume_MA10'] = data['Volume'].rolling(window=10).mean()
            
            stock_ret = data['Close'].pct_change().rolling(rpi_period).sum()
            nifty_ret_aligned = nifty_data.pct_change().rolling(rpi_period).sum()
            data['RPI'] = (1 + stock_ret) / (1 + nifty_ret_aligned).replace(0, np.nan)
            
            latest = data.iloc[-1]
            stock_data_dict = {
                'symbol': symbol, 'frequency': frequency, 'current_price': latest['Close'],
                'adx': latest.get('ADX_14', 0), 'rsi': latest.get('RSI_14', 0), 'rpi': latest.get('RPI', 0),
                'volume_ratio': latest['Volume'] / latest['Volume_MA10'] if latest.get('Volume_MA10', 0) > 0 else 0,
                'is_filtered': bool(
                    latest.get('ADX_14', 0) > 25 and latest.get('RSI_14', 0) > 55 and
                    latest.get('RPI', 0) > 1 and latest['Close'] > latest.get('EMA_11', 0) > latest.get('EMA_21', 0) > latest.get('EMA_40', 0)
                )
            }
            all_stocks_data.append(stock_data_dict)

        if not all_stocks_data:
            logger.info("No stocks could be processed.")
            return

        df = pd.DataFrame(all_stocks_data)
        filtered_df = df[df['is_filtered']].copy()

        if not filtered_df.empty:
            close_prices = all_data.loc[:, (filtered_df['symbol'].tolist(), 'Close')]
            three_month_returns = (close_prices.iloc[-1] / close_prices.iloc[-quarter_lookback] - 1) * 100
            three_month_returns.name = '3m_change_pct'
            
            filtered_df = filtered_df.merge(three_month_returns, left_on='symbol', right_index=True)
            filtered_df['rs_rating'] = filtered_df['3m_change_pct'].rank(pct=True) * 100
            
            filtered_df['rank'] = filtered_df['rs_rating'].rank(ascending=False)
            df = df.merge(filtered_df[['symbol', 'rank']], on='symbol', how='left')

        db = sqlite3.connect(DB_PATH)
        cursor = db.cursor()
        cursor.execute("DELETE FROM screener_results WHERE frequency = ?", (frequency,))
        df.to_sql('screener_results', db, if_exists='append', index=False)
        db.close()
        logger.info(f"Screener process finished. Saved {len(df)} stocks for {frequency} to the database.")
    except Exception as e:
        logger.error(f"A critical error occurred during the screener process for {frequency}: {e}", exc_info=True)

if __name__ == '__main__':
    run_screener_process('weekly')
