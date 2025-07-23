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
        # Check if today is a weekday and the market has already closed.
        market_has_closed_for_today = now.weekday() < 5 and now.time() >= market_close_time

        if market_has_closed_for_today:
            # If run after market close on a weekday, it's safe to use today's data.
            return today
        else:
            # Otherwise, the market is open, hasn't opened, or it's a weekend.
            # We need to find the last fully completed trading day.
            if now.weekday() == 0:  # It's Monday before market close
                return today - timedelta(days=3)  # Use last Friday's data
            elif now.weekday() >= 5:  # It's Saturday or Sunday
                return today - timedelta(days=now.weekday() - 4)  # Use last Friday's data
            else:  # It's Tuesday-Friday before market close
                return today - timedelta(days=1)  # Use yesterday's data

    elif frequency == 'weekly':
        today_weekday = now.weekday()
        # If it's before Friday's market close, we must use the previous week's data.
        if today_weekday < 4 or (today_weekday == 4 and now.time() < market_close_time):
            last_monday = today - timedelta(days=today_weekday)
            return last_monday - timedelta(days=3)
        else:
            # If it's after market close on Friday, or on a weekend, use today's date.
            # yfinance will fetch data up to the most recently completed week.
            return today

    elif frequency == 'monthly':
        first_day_of_current_month = today.replace(day=1)
        last_day_of_previous_month = first_day_of_current_month - timedelta(days=1)
        _, last_day_num = calendar.monthrange(today.year, today.month)
        is_last_day_of_month = (today.day == last_day_num)

        # It is safe to use the current month's data only if it's run *after* the market
        # has closed on the *last day* of the current month.
        if is_last_day_of_month and now.time() >= market_close_time:
            return today
        else:
            return last_day_of_previous_month
    
    return today # Fallback

def get_nse_stocks_from_db():
    """
    Fetches the list of stock symbols from the master_stocks table in the database.
    """
    print("Fetching stock list from the database...")
    db = sqlite3.connect(DB_PATH)
    try:
        symbols_df = pd.read_sql_query("SELECT symbol FROM master_stocks", db)
        symbols = symbols_df['symbol'].tolist()
        logger.info(f"Found {len(symbols)} stocks in the master list.")
        return symbols
    except Exception as e:
        logger.error(f"Error fetching stocks from database: {e}")
        return []
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

    return nearest_support if pd.notna(nearest_support) else 0, \
           nearest_resistance if pd.notna(nearest_resistance) else 0

def get_crossover_date(data):
    """Finds the actual date of the last EMA 11/21 crossover event."""
    cross_series = data['EMA_11'] > data['EMA_21']
    if not cross_series.iloc[-1]: # No current crossover
        return None

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
        # Add 1 day to the end date for yfinance because it's exclusive
        end_date_exclusive = end_date_inclusive + timedelta(days=1)

        logger.info(f"Running screener for {frequency} with data ending on {end_date_inclusive.strftime('%Y-%m-%d')}")


        nifty_data = yf.download('^NSEI', start=start_date, end=end_date_exclusive, interval=interval, progress=False)
        if isinstance(nifty_data.columns, pd.MultiIndex):
            nifty_data.columns = nifty_data.columns.droplevel(1)

        all_stocks_data = []
        total_stocks = len(all_symbols)

        for i, symbol in enumerate(all_symbols):
            logger.info(f"Processing {i+1}/{total_stocks}: {symbol} ({frequency})")
            try:
                data = yf.download(symbol, start=start_date, end=end_date_exclusive, interval=interval, progress=False)

                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)

                if data.empty or len(data) < 50:
                    continue

                data = calculate_indicators(data)

                stock_ret = data['Close'].pct_change().rolling(rpi_period).sum()
                nifty_ret_aligned = nifty_data['Close'].pct_change().rolling(rpi_period).sum()
                data['RPI'] = stock_ret / nifty_ret_aligned

                latest = data.iloc[-1]

                support, resistance = find_pivots(data)
                crossover_date = get_crossover_date(data)

                avg_move_4_periods = data['Close'].pct_change().tail(4).mean()
                avg_move_52_periods = data['Close'].pct_change().tail(52).mean()
                momentum_burst_score = (avg_move_4_periods / avg_move_52_periods) if avg_move_52_periods != 0 else 0

                stock_data_dict = {
                    'symbol': symbol,
                    'frequency': frequency,
                    'current_price': latest['Close'],
                    'crossover_date': crossover_date,
                    'adx': latest.get('ADX_14', 0),
                    'rsi': latest.get('RSI_14', 0),
                    'rpi': latest.get('RPI', 0),
                    'volume_ratio': latest['Volume'] / latest['Volume_MA10'] if latest.get('Volume_MA10', 0) > 0 else 0,
                    'support': support,
                    'resistance': resistance,
                    'dist_ema11_pct': ((latest['Close'] - latest.get('EMA_11', 0)) / latest.get('EMA_11', 0)) * 100 if latest.get('EMA_11', 0) > 0 else 0,
                    'dist_ema21_pct': ((latest['Close'] - latest.get('EMA_21', 0)) / latest.get('EMA_21', 0)) * 100 if latest.get('EMA_21', 0) > 0 else 0,
                    'fifty_two_week_low': data['Low'].rolling(year_lookback).min().iloc[-1],
                    'fifty_two_week_high': data['High'].rolling(year_lookback).max().iloc[-1],
                    '3m_change_pct': (data['Close'].iloc[-1] / data['Close'].iloc[-quarter_lookback] - 1) * 100 if len(data) > quarter_lookback else 0,
                    'crossover_weeks_ago': (date.today() - datetime.strptime(crossover_date, '%Y-%m-%d').date()).days // (7 if frequency == 'weekly' else (30 if frequency == 'monthly' else 1)) if crossover_date else 999,
                    'atr': latest.get('ATRr_14', 0),
                    'momentum_burst_score': momentum_burst_score
                }

                is_ema40_sloping_up_short = data['EMA_40'].iloc[-1] > data['EMA_40'].iloc[-5]

                stock_data_dict['is_filtered'] = bool(
                    stock_data_dict['adx'] > 25 and
                    stock_data_dict['rsi'] > 55 and
                    stock_data_dict['rpi'] > 1 and
                    latest['Close'] > latest.get('EMA_11', 0) > latest.get('EMA_21', 0) > latest.get('EMA_40', 0) and
                    stock_data_dict['volume_ratio'] > 1.25 and
                    is_ema40_sloping_up_short
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
            filtered_df['rs_rating'] = filtered_df['3m_change_pct'].rank(pct=True) * 100

            filtered_df['near_52w_high_score'] = filtered_df['current_price'] / filtered_df['fifty_two_week_high']

            filtered_df['crossover_score'] = np.select([filtered_df['crossover_weeks_ago'] <= 3, filtered_df['crossover_weeks_ago'] <= 6], [1.0, 0.8], default=0.5)
            filtered_df['volatility_score'] = 1 / (filtered_df['atr'] + 1)
            filtered_df['rs_rating_score'] = filtered_df['rs_rating'] / 100
            filtered_df['volume_score'] = np.clip(df['volume_ratio'] - 1, 0, 1)

            weights = {'crossover': 0.20, 'rs': 0.25, 'volume': 0.1, '52w_high': 0.20, 'momentum': 0.25}

            filtered_df['final_rank_score'] = (
                filtered_df['crossover_score'] * weights['crossover'] +
                filtered_df['rs_rating_score'] * weights['rs'] +
                filtered_df['volume_score'] * weights['volume'] +
                filtered_df['near_52w_high_score'] * weights['52w_high'] +
                filtered_df['momentum_burst_score'].fillna(0) * weights['momentum']
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
