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
                if len(chunk) == 1:
                     data_chunk.columns = pd.MultiIndex.from_product([data_chunk.columns, chunk])
                all_data = pd.concat([all_data, data_chunk], axis=1)
        except Exception as e:
            logger.error(f"Failed to download chunk {i//chunk_size + 1}: {e}")
    return all_data

def find_pivots(data, lookback=10):
    lows = data['Low'].rolling(window=lookback*2+1, center=True).min()
    highs = data['High'].rolling(window=lookback*2+1, center=True).max()
    support = lows[lows == data['Low']].dropna()
    resistance = highs[highs == data['High']].dropna()
    current_price = data['Close'].iloc[-1]
    nearest_support = support[support < current_price].max()
    nearest_resistance = resistance[resistance > current_price].min()
    return nearest_support if pd.notna(nearest_support) else 0, nearest_resistance if pd.notna(nearest_resistance) else 0

def get_crossover_date(data):
    cross_series = data['EMA_11'] > data['EMA_21']
    if not cross_series.iloc[-1]: return None
    crossover_points = cross_series.astype(int).diff()
    last_crossover_start_index = crossover_points[crossover_points == 1].index.max()
    if pd.isna(last_crossover_start_index):
        return cross_series.index[0].strftime('%Y-%m-%d')
    return last_crossover_start_index.strftime('%Y-%m-%d')


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
        
        logger.info("Fetching benchmark data for ^NSEI...")
        nifty_data_df = yf.download('^NSEI', start=start_date, end=end_date_exclusive, interval=interval, auto_adjust=True)
        if nifty_data_df.empty:
            logger.error("Failed to download benchmark data (^NSEI). Aborting screener.")
            return
        nifty_data = nifty_data_df['Close']

        logger.info(f"Fetching data for {len(all_symbols)} stocks in chunks...")
        all_data = download_data_in_chunks(all_symbols, start_date, end_date_exclusive, interval)
        if all_data.empty:
            logger.error("Failed to download any stock data. Aborting screener.")
            return
        
        all_data.columns = all_data.columns.swaplevel(0, 1)

        nifty_ret = nifty_data.pct_change().rolling(rpi_period).sum()

        all_stocks_data = []
        for symbol in all_symbols:
            if symbol not in all_data.columns.get_level_values(0): continue
            data = all_data[symbol].copy()
            
            # FIX: Drop rows with missing close prices to prevent NaN issues
            data.dropna(subset=['Close'], inplace=True)

            if data.empty or len(data) < 50: continue

            data.ta.adx(length=14, append=True)
            data.ta.rsi(length=14, append=True)
            data.ta.ema(length=11, append=True)
            data.ta.ema(length=21, append=True)
            data.ta.ema(length=40, append=True)
            
            data['Volume_MA10'] = data['Volume'].rolling(window=10, min_periods=1).mean()
            
            stock_ret_series = data['Close'].pct_change().rolling(rpi_period).sum()
            
            if stock_ret_series.empty: continue
            
            latest_stock_ret = stock_ret_series.values[-1]
            
            nifty_ret_aligned = nifty_ret.reindex(data.index, method='ffill')
            if nifty_ret_aligned.empty: continue
            latest_nifty_ret = nifty_ret_aligned.values[-1]
            
            rpi_value = np.nan
            denominator = 1 + latest_nifty_ret

            if pd.notna(latest_stock_ret) and pd.notna(denominator) and denominator != 0:
                rpi_value = (1 + latest_stock_ret) / denominator
            
            latest = data.iloc[-1]
            previous = data.iloc[-2]
            print(latest, previous)

            is_making_higher_close = latest['Close'] > previous['Close']
            
            ema_40_val = latest.get('EMA_40')
            ema_40_prev_val = data['EMA_40'].iloc[-6] if len(data) >= 6 and pd.notna(data['EMA_40'].iloc[-6]) else None
            is_ema40_sloping_up = ema_40_val > ema_40_prev_val if pd.notna(ema_40_val) and pd.notna(ema_40_prev_val) else False

            adx_val = float(latest.get('ADX_14', 0)) if pd.notna(latest.get('ADX_14')) else None
            rsi_val = float(latest.get('RSI_14', 0)) if pd.notna(latest.get('RSI_14')) else None
            rpi_val = float(rpi_value) if pd.notna(rpi_value) else None
            
            volume_ma = latest.get('Volume_MA10')
            volume_ratio_val = 0.0
            if pd.notna(volume_ma) and volume_ma > 0:
                volume_ratio_val = float(latest['Volume'] / volume_ma)

            is_filtered = bool(
                adx_val is not None and adx_val > 25 and
                rsi_val is not None and rsi_val > 55 and
                rpi_val is not None and rpi_val > 1 and 
                volume_ratio_val > 1.25 and
                latest['Close'] > latest.get('EMA_11', 0) > latest.get('EMA_21', 0) > ema_40_val and
                is_ema40_sloping_up and
                is_making_higher_close
            )

            support, resistance = find_pivots(data)
            crossover_date = get_crossover_date(data)
            dist_ema11_pct = ((latest['Close'] - latest.get('EMA_11', 0)) / latest.get('EMA_11', 0)) * 100 if latest.get('EMA_11', 0) > 0 else 0
            dist_ema21_pct = ((latest['Close'] - latest.get('EMA_21', 0)) / latest.get('EMA_21', 0)) * 100 if latest.get('EMA_21', 0) > 0 else 0
            fifty_two_week_low = data['Low'].rolling(year_lookback).min().iloc[-1]
            fifty_two_week_high = data['High'].rolling(year_lookback).max().iloc[-1]

            stock_data_dict = {
                'symbol': symbol, 'frequency': frequency, 'current_price': latest['Close'],
                'adx': adx_val, 'rsi': rsi_val, 'rpi': rpi_val, 'volume_ratio': volume_ratio_val,
                'crossover_date': crossover_date,
                'support': float(support) if pd.notna(support) else None,
                'resistance': float(resistance) if pd.notna(resistance) else None,
                'dist_ema11_pct': float(dist_ema11_pct) if pd.notna(dist_ema11_pct) else None,
                'dist_ema21_pct': float(dist_ema21_pct) if pd.notna(dist_ema21_pct) else None,
                'fifty_two_week_low': float(fifty_two_week_low) if pd.notna(fifty_two_week_low) else None,
                'fifty_two_week_high': float(fifty_two_week_high) if pd.notna(fifty_two_week_high) else None,
                'is_filtered': is_filtered,
                'ema_11': float(latest.get('EMA_11')) if pd.notna(latest.get('EMA_11')) else None,
                'ema_21': float(latest.get('EMA_21')) if pd.notna(latest.get('EMA_21')) else None,
                'ema_40': float(ema_40_val) if pd.notna(ema_40_val) else None,
                'ema_40_prev': float(ema_40_prev_val) if pd.notna(ema_40_prev_val) else None,
                'prev_close': float(previous['Close']) if pd.notna(previous['Close']) else None
            }
            all_stocks_data.append(stock_data_dict)

        if not all_stocks_data:
            logger.info("No stocks could be processed.")
            return

        df = pd.DataFrame(all_stocks_data)
        filtered_df = df[df['is_filtered']].copy()

        if not filtered_df.empty:
            valid_symbols = [s for s in filtered_df['symbol'].tolist() if s in all_data.columns.get_level_values(0)]
            close_prices = all_data.loc[:, (valid_symbols, 'Close')]
            
            three_month_returns = (close_prices.iloc[-1] / close_prices.iloc[-quarter_lookback] - 1) * 100
            three_month_returns.name = '3m_change_pct'
            
            if isinstance(three_month_returns.index, pd.MultiIndex):
                three_month_returns.index = three_month_returns.index.get_level_values(0)
            
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
