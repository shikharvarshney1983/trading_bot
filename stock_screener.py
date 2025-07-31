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
# Make sure logging is configured if running standalone
if not logger.handlers:
    logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Configuration for different timeframes
TIME_CONFIGS = {
    'daily': {
        'interval': '1d', 'history_days': 365 * 2, 'rpi_period': 50,
        'year_lookback': 252, 'quarter_lookback': 63,
    },
    'weekly': {
        'interval': '1wk', 'history_days': 365 * 5, 'rpi_period': 20,
        'year_lookback': 52, 'quarter_lookback': 13,
    },
    'monthly': {
        'interval': '1mo', 'history_days': 365 * 10, 'rpi_period': 12,
        'year_lookback': 12, 'quarter_lookback': 3,
    }
}

def get_screener_end_date(frequency):
    tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(tz)
    today = now.date()
    market_close_time = datetime(2000, 1, 1, 15, 30).time()
    if frequency == 'daily':
        market_has_closed_for_today = now.weekday() < 5 and now.time() >= market_close_time
        if market_has_closed_for_today: return today
        else:
            if now.weekday() == 0: return today - timedelta(days=3)
            elif now.weekday() >= 5: return today - timedelta(days=now.weekday() - 4)
            else: return today - timedelta(days=1)
    elif frequency == 'weekly':
        today_weekday = now.weekday()
        if today_weekday < 4 or (today_weekday == 4 and now.time() < market_close_time):
            last_monday = today - timedelta(days=today_weekday)
            return last_monday - timedelta(days=3)
        else: return today
    elif frequency == 'monthly':
        first_day_of_current_month = today.replace(day=1)
        last_day_of_previous_month = first_day_of_current_month - timedelta(days=1)
        _, last_day_num = calendar.monthrange(today.year, today.month)
        is_last_day_of_month = (today.day == last_day_num)
        if is_last_day_of_month and now.time() >= market_close_time: return today
        else: return last_day_of_previous_month
    return today

def get_nse_stocks_from_db():
    db = sqlite3.connect(DB_PATH)
    try:
        symbols_df = pd.read_sql_query("SELECT symbol FROM master_stocks", db)
        return symbols_df['symbol'].tolist()
    finally:
        db.close()

def calculate_indicators(data):
    data.ta.adx(length=14, append=True)
    data.ta.rsi(length=14, append=True)
    data.ta.ema(length=11, append=True)
    data.ta.ema(length=21, append=True)
    data.ta.ema(length=40, append=True)
    data.ta.atr(length=14, append=True)
    data['Volume_MA10'] = data['Volume'].rolling(window=10).mean()
    return data

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
        logger.info(f"Fetching data for {len(all_symbols)} stocks in one batch...")

        all_data = yf.download(all_symbols + ['^NSEI'], start=start_date, end=end_date_exclusive, interval=interval, progress=True)
        
        if all_data.empty:
            logger.error("Failed to download any stock data. Aborting screener.")
            return
            
        nifty_data = all_data['Close']['^NSEI']
        all_stocks_data = []

        for i, symbol in enumerate(all_symbols):
            try:
                # FIX: Use .copy() to avoid SettingWithCopyWarning
                data = all_data.loc[:, (slice(None), symbol)].copy()
                data.columns = data.columns.droplevel(1)
                
                if data.empty or data['Close'].isnull().all() or len(data) < 50:
                    continue

                data = calculate_indicators(data)
                stock_ret = data['Close'].pct_change().rolling(rpi_period).sum()
                nifty_ret_aligned = nifty_data.pct_change().rolling(rpi_period).sum()
                denominator = 1 + nifty_ret_aligned
                data['RPI'] = (1 + stock_ret) / denominator.replace(0, np.nan)

                latest, previous = data.iloc[-1], data.iloc[-2]
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
                    stock_data_dict['adx'] > 25 and stock_data_dict['rsi'] > 55 and
                    stock_data_dict['rpi'] > 1 and latest['Close'] > latest.get('EMA_11', 0) > latest.get('EMA_21', 0) > latest.get('EMA_40', 0) and
                    stock_data_dict['volume_ratio'] > 1.25 and is_ema40_sloping_up_short and
                    is_making_higher_close and is_not_major_reversal
                )
                all_stocks_data.append(stock_data_dict)
            except Exception as e:
                logger.error(f"Could not process {symbol} from bulk download: {e}")
                continue

        if not all_stocks_data:
            logger.info("No stocks could be processed after analysis.")
            return

        df = pd.DataFrame(all_stocks_data)
        filtered_df = df[df['is_filtered']].copy()

        if not filtered_df.empty:
            # FIX: Calculate 3m_change_pct efficiently from the bulk data
            changes = []
            for symbol in filtered_df['symbol']:
                stock_data = all_data.loc[:, (slice(None), symbol)]
                stock_data.columns = stock_data.columns.droplevel(1)
                
                if len(stock_data) > quarter_lookback:
                    change = (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[-quarter_lookback] - 1) * 100
                    changes.append(change)
                else:
                    changes.append(0)
            
            filtered_df['3m_change_pct'] = changes
            
            filtered_df['rs_rating'] = filtered_df['3m_change_pct'].rank(pct=True) * 100
            filtered_df['near_52w_high_score'] = filtered_df['current_price'] / filtered_df['fifty_two_week_high']
            filtered_df['crossover_weeks_ago'] = filtered_df.apply(lambda row: (date.today() - datetime.strptime(row['crossover_date'], '%Y-%m-%d').date()).days // (7 if frequency == 'weekly' else (30 if frequency == 'monthly' else 1)) if row['crossover_date'] else 999, axis=1)
            filtered_df['crossover_score'] = np.select([filtered_df['crossover_weeks_ago'] <= 3, filtered_df['crossover_weeks_ago'] <= 6], [1.0, 0.8], default=0.5)
            filtered_df['volume_score'] = np.clip(filtered_df['volume_ratio'] - 1, 0, 1)
            weights = {'crossover': 0.25, 'rs': 0.35, 'volume': 0.15, '52w_high': 0.25}
            filtered_df['final_rank_score'] = (
                filtered_df['crossover_score'] * weights['crossover'] + (filtered_df['rs_rating'] / 100) * weights['rs'] +
                filtered_df['volume_score'] * weights['volume'] + filtered_df['near_52w_high_score'] * weights['52w_high']
            )
            filtered_df = filtered_df.sort_values('final_rank_score', ascending=False).reset_index(drop=True)
            filtered_df['rank'] = filtered_df.index + 1
            df = df.merge(filtered_df[['symbol', 'rank']], on='symbol', how='left')

        db = sqlite3.connect(DB_PATH)
        cursor = db.cursor()
        cursor.execute("DELETE FROM screener_results WHERE frequency = ?", (frequency,))
        # Use to_sql for efficient bulk insertion
        df.to_sql('screener_results', db, if_exists='append', index=False)
        db.close()
        logger.info(f"Screener process finished. Saved {len(df)} stocks for {frequency} to the database.")
    except Exception as e:
        logger.error(f"A critical error occurred during the screener process for {frequency}: {e}", exc_info=True)

if __name__ == '__main__':
    run_screener_process('daily')
