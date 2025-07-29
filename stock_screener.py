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

DB_PATH = os.path.join('instance', 'trading.db')
logger = logging.getLogger(__name__)
logging.basicConfig(filename='trading_bot.log', level=logging.INFO)

# Configuration for different timeframes
TIME_CONFIGS = {
    'daily': {
        'interval': '1d',
        'history_days': 365 * 2,
        'rpi_period': 50,
        'year_lookback': 252,
        'quarter_lookback': 63,
    },
    'weekly': {
        'interval': '1wk',
        'history_days': 365 * 5,
        'rpi_period': 20,
        'year_lookback': 52,
        'quarter_lookback': 13,
    },
    'monthly': {
        'interval': '1mo',
        'history_days': 365 * 10,
        'rpi_period': 12,
        'year_lookback': 12,
        'quarter_lookback': 3,
    }
}

def get_screener_end_date(frequency):
    """
    Determines the correct end date for analysis to avoid using partial data
    when the screener is run during market hours.
    """
    tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(tz)
    today = now.date()
    market_close_time = datetime(2000, 1, 1, 15, 30).time()

    if frequency == 'daily':
        market_has_closed_for_today = now.weekday() < 5 and now.time() >= market_close_time
        if market_has_closed_for_today:
            return today
        else:
            if now.weekday() == 0:
                return today - timedelta(days=3)
            elif now.weekday() >= 5:
                return today - timedelta(days=now.weekday() - 4)
            else:
                return today - timedelta(days=1)

    elif frequency == 'weekly':
        today_weekday = now.weekday()
        if today_weekday < 4 or (today_weekday == 4 and now.time() < market_close_time):
            last_monday = today - timedelta(days=today_weekday)
            return last_monday - timedelta(days=3)
        else:
            return today

    elif frequency == 'monthly':
        first_day_of_current_month = today.replace(day=1)
        last_day_of_previous_month = first_day_of_current_month - timedelta(days=1)
        _, last_day_num = calendar.monthrange(today.year, today.month)
        is_last_day_of_month = (today.day == last_day_num)
        if is_last_day_of_month and now.time() >= market_close_time:
            return today
        else:
            return last_day_of_previous_month
    
    return today

def get_nse_stocks_from_db():
    """
    Fetches the list of stock symbols from the master_stocks table in the database.
    """
    db = sqlite3.connect(DB_PATH)
    try:
        symbols_df = pd.read_sql_query("SELECT symbol FROM master_stocks", db)
        return symbols_df['symbol'].tolist()
    finally:
        db.close()

def calculate_indicators(data):
    """Calculates all necessary technical indicators for a given stock data DataFrame."""
    data.ta.adx(length=14, append=True)
    data.ta.rsi(length=14, append=True)
    data.ta.ema(length=11, append=True)
    data.ta.ema(length=21, append=True)
    data.ta.ema(length=40, append=True)
    data.ta.atr(length=14, append=True)
    data['Volume_MA10'] = data['Volume'].rolling(window=10).mean()
    return data

def find_pivots(data, lookback=10):
    """Finds the nearest support and resistance levels."""
    lows = data['Low'].rolling(window=lookback*2+1, center=True).min()
    highs = data['High'].rolling(window=lookback*2+1, center=True).max()
    support = lows[lows == data['Low']].dropna()
    resistance = highs[highs == data['High']].dropna()
    current_price = data['Close'].iloc[-1]
    nearest_support = support[support < current_price].max()
    nearest_resistance = resistance[resistance > current_price].min()
    return nearest_support if pd.notna(nearest_support) else 0, nearest_resistance if pd.notna(nearest_resistance) else 0

def get_crossover_date(data):
    """Finds the actual date of the last EMA 11/21 crossover event."""
    cross_series = data['EMA_11'] > data['EMA_21']
    if not cross_series.iloc[-1]: return None
    crossover_points = cross_series.astype(int).diff()
    last_crossover_start_index = crossover_points[crossover_points == 1].index.max()
    if pd.isna(last_crossover_start_index):
        return cross_series.index[0].strftime('%Y-%m-%d')
    return last_crossover_start_index.strftime('%Y-%m-%d')

def run_screener_process(frequency='weekly'):
    """Main function to run the entire screening and ranking process."""
    logger.info(f"Starting {frequency} stock screener process...")
    try:
        config = TIME_CONFIGS[frequency]
        interval = config['interval']
        history_days = config['history_days']
        rpi_period = config['rpi_period']
        year_lookback = config['year_lookback']
        quarter_lookback = config['quarter_lookback']

        all_symbols = get_nse_stocks_from_db()
        if not all_symbols:
            logger.info("No stocks in master list. Aborting.")
            return

        end_date_inclusive = get_screener_end_date(frequency)
        start_date = end_date_inclusive - timedelta(days=history_days)
        end_date_exclusive = end_date_inclusive + timedelta(days=1)
        logger.info(f"Running screener for {frequency} with data ending on {end_date_inclusive.strftime('%Y-%m-%d')}")

        nifty_data = yf.download('^NSEI', start=start_date, end=end_date_exclusive, interval=interval, progress=False)
        if isinstance(nifty_data.columns, pd.MultiIndex):
            nifty_data.columns = nifty_data.columns.droplevel(1)

        all_stocks_data = []
        for i, symbol in enumerate(all_symbols):
            logger.info(f"Processing {i+1}/{len(all_symbols)}: {symbol} ({frequency})")
            try:
                data = yf.download(symbol, start=start_date, end=end_date_exclusive, interval=interval, progress=False)
                
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)

                if data.empty or len(data) < 50: continue

                data = calculate_indicators(data)
                stock_ret = data['Close'].pct_change().rolling(rpi_period).sum()
                nifty_ret_aligned = nifty_data['Close'].pct_change().rolling(rpi_period).sum()
                
                # --- FIX: Robust RPI Calculation ---
                # Calculate RPI using the robust ratio method to handle negative returns correctly
                denominator = 1 + nifty_ret_aligned
                # Use .replace(0, np.nan) to prevent division by zero errors
                data['RPI'] = (1 + stock_ret) / denominator.replace(0, np.nan)


                latest = data.iloc[-1]
                previous = data.iloc[-2]

                support, resistance = find_pivots(data)
                crossover_date = get_crossover_date(data)

                stock_data_dict = {
                    'symbol': symbol, 'frequency': frequency, 'current_price': latest['Close'],
                    'crossover_date': crossover_date, 'adx': latest.get('ADX_14', 0),
                    'rsi': latest.get('RSI_14', 0), 'rpi': latest.get('RPI', 0),
                    'volume_ratio': latest['Volume'] / latest['Volume_MA10'] if latest.get('Volume_MA10', 0) > 0 else 0,
                    'support': support, 'resistance': resistance,
                    'dist_ema11_pct': ((latest['Close'] - latest.get('EMA_11', 0)) / latest.get('EMA_11', 0)) * 100 if latest.get('EMA_11', 0) > 0 else 0,
                    'dist_ema21_pct': ((latest['Close'] - latest.get('EMA_21', 0)) / latest.get('EMA_21', 0)) * 100 if latest.get('EMA_21', 0) > 0 else 0,
                    'fifty_two_week_low': data['Low'].rolling(year_lookback).min().iloc[-1],
                    'fifty_two_week_high': data['High'].rolling(year_lookback).max().iloc[-1],
                }

                is_making_higher_close = latest['Close'] > previous['Close']
                intraday_move_pct = ((latest['Close'] - latest['Open']) / latest['Open']) * 100
                is_not_major_reversal = intraday_move_pct > -1.0
                is_ema40_sloping_up_short = data['EMA_40'].iloc[-1] > data['EMA_40'].iloc[-5]

                stock_data_dict['is_filtered'] = bool(
                    stock_data_dict['adx'] > 25 and
                    stock_data_dict['rsi'] > 55 and
                    stock_data_dict['rpi'] > 1 and
                    latest['Close'] > latest.get('EMA_11', 0) > latest.get('EMA_21', 0) > latest.get('EMA_40', 0) and
                    stock_data_dict['volume_ratio'] > 1.25 and
                    is_ema40_sloping_up_short and
                    is_making_higher_close and
                    is_not_major_reversal
                )
                all_stocks_data.append(stock_data_dict)
            except Exception as e:
                logger.error(f"Could not process {symbol}: {e}")
                continue

        if not all_stocks_data:
            logger.info("No stocks could be processed.")
            return

        df = pd.DataFrame(all_stocks_data)
        filtered_df = df[df['is_filtered']].copy()

        if not filtered_df.empty:
            filtered_df['3m_change_pct'] = filtered_df.apply(lambda row: (row['current_price'] / data['Close'].iloc[-quarter_lookback] - 1) * 100 if len(data) > quarter_lookback else 0, axis=1)
            filtered_df['rs_rating'] = filtered_df['3m_change_pct'].rank(pct=True) * 100
            filtered_df['near_52w_high_score'] = filtered_df['current_price'] / filtered_df['fifty_two_week_high']
            filtered_df['crossover_weeks_ago'] = filtered_df.apply(lambda row: (date.today() - datetime.strptime(row['crossover_date'], '%Y-%m-%d').date()).days // (7 if frequency == 'weekly' else (30 if frequency == 'monthly' else 1)) if row['crossover_date'] else 999, axis=1)
            filtered_df['crossover_score'] = np.select([filtered_df['crossover_weeks_ago'] <= 3, filtered_df['crossover_weeks_ago'] <= 6], [1.0, 0.8], default=0.5)
            filtered_df['volume_score'] = np.clip(filtered_df['volume_ratio'] - 1, 0, 1)
            
            weights = {'crossover': 0.25, 'rs': 0.35, 'volume': 0.15, '52w_high': 0.25}
            filtered_df['final_rank_score'] = (
                filtered_df['crossover_score'] * weights['crossover'] +
                (filtered_df['rs_rating'] / 100) * weights['rs'] +
                filtered_df['volume_score'] * weights['volume'] +
                filtered_df['near_52w_high_score'] * weights['52w_high']
            )
            filtered_df = filtered_df.sort_values('final_rank_score', ascending=False).reset_index(drop=True)
            filtered_df['rank'] = filtered_df.index + 1
            df = df.merge(filtered_df[['symbol', 'rank']], on='symbol', how='left')

        db = sqlite3.connect(DB_PATH)
        cursor = db.cursor()
        cursor.execute("DELETE FROM screener_results WHERE frequency = ?", (frequency,))
        for _, row in df.iterrows():
            cursor.execute(
                """INSERT INTO screener_results (
                    symbol, frequency, current_price, crossover_date, adx, rsi, rpi, volume_ratio,
                    support, resistance, dist_ema11_pct, dist_ema21_pct,
                    fifty_two_week_low, fifty_two_week_high, rank, is_filtered
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row['symbol'], row['frequency'], row['current_price'], row['crossover_date'],
                    row['adx'], row['rsi'], row['rpi'], row['volume_ratio'],
                    row['support'], row['resistance'], row['dist_ema11_pct'], row['dist_ema21_pct'],
                    row['fifty_two_week_low'], row['fifty_two_week_high'],
                    row.get('rank'), row['is_filtered']
                )
            )
        db.commit()
        db.close()
        logger.info(f"Screener process finished. Saved {len(df)} stocks for {frequency} to the database.")
    except Exception as e:
        logger.error(f"A critical error occurred during the screener process for {frequency}: {e}")

if __name__ == '__main__':
    run_screener_process('weekly')
