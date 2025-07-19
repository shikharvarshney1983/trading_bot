# stock_screener.py
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from datetime import date, timedelta, datetime
import sqlite3
import numpy as np
import os

DB_PATH = os.path.join('instance', 'trading.db')

def get_nse_stocks_from_db():
    """
    Fetches the list of stock symbols from the master_stocks table in the database.
    """
    print("Fetching stock list from the database...")
    db = sqlite3.connect(DB_PATH)
    try:
        symbols_df = pd.read_sql_query("SELECT symbol FROM master_stocks", db)
        symbols = symbols_df['symbol'].tolist()
        print(f"Found {len(symbols)} stocks in the master list.")
        return symbols
    except Exception as e:
        print(f"Error fetching stocks from database: {e}")
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
    
    # A change from False to True is a diff of 1 (True - False = 1)
    crossover_points = cross_series.astype(int).diff()
    
    # Find the last time a crossover started
    last_crossover_start_index = crossover_points[crossover_points == 1].index.max()
    
    if pd.isna(last_crossover_start_index):
        # This means it has been in a crossover state for the whole period
        return cross_series.index[0].strftime('%Y-%m-%d')
        
    return last_crossover_start_index.strftime('%Y-%m-%d')

def run_screener_process():
    """Main function to run the entire screening and ranking process."""
    print("Starting stock screener process...")
    try:
        all_symbols = get_nse_stocks_from_db()
        if not all_symbols:
            print("No stocks in master list. Aborting.")
            return

        nifty_data = yf.download('^NSEI', period='2y', interval='1wk')
        if isinstance(nifty_data.columns, pd.MultiIndex):
            nifty_data.columns = nifty_data.columns.droplevel(1)

        all_stocks_data = []
        total_stocks = len(all_symbols)

        for i, symbol in enumerate(all_symbols):
            print(f"Processing {i+1}/{total_stocks}: {symbol}")
            try:
                data = yf.download(symbol, period='2y', interval='1wk', progress=False)
                
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)

                if data.empty or len(data) < 100:
                    continue

                data = calculate_indicators(data)
                
                roll_period = 50
                stock_ret = data['Close'].pct_change().rolling(roll_period).sum()
                nifty_ret_aligned = nifty_data['Close'].pct_change().rolling(roll_period).sum()
                data['RPI'] = stock_ret / nifty_ret_aligned

                latest = data.iloc[-1]
                
                support, resistance = find_pivots(data)
                crossover_date = get_crossover_date(data)
                
                # Pre-calculate momentum burst score to fix error and improve performance
                avg_move_4w = data['Close'].pct_change().tail(4).mean()
                avg_move_52w = data['Close'].pct_change().tail(52).mean()
                momentum_burst_score = (avg_move_4w / avg_move_52w) if avg_move_52w != 0 else 0
                
                stock_data_dict = {
                    'symbol': symbol,
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
                    'fifty_two_week_low': data['Low'].rolling(52).min().iloc[-1],
                    'fifty_two_week_high': data['High'].rolling(52).max().iloc[-1],
                    '3m_change_pct': (data['Close'].iloc[-1] / data['Close'].iloc[-13] - 1) * 100 if len(data) > 13 else 0,
                    'crossover_weeks_ago': (date.today() - datetime.strptime(crossover_date, '%Y-%m-%d').date()).days // 7 if crossover_date else 99,
                    'atr': latest.get('ATRr_14', 0),
                    'momentum_burst_score': momentum_burst_score
                }
                
                # Filtering criteria
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
                print(f"Could not process {symbol}: {e}")
                continue

        if not all_stocks_data:
            print("No stocks could be processed.")
            return

        df = pd.DataFrame(all_stocks_data)
        
        # Ranking only applies to filtered stocks
        filtered_df = df[df['is_filtered']].copy()
        if not filtered_df.empty:
            filtered_df['rs_rating'] = filtered_df['3m_change_pct'].rank(pct=True) * 100
            
            # New ranking criteria using the pre-calculated score
            filtered_df['near_52w_high_score'] = filtered_df['current_price'] / filtered_df['fifty_two_week_high']
            
            filtered_df['crossover_score'] = np.select([filtered_df['crossover_weeks_ago'] <= 3, filtered_df['crossover_weeks_ago'] <= 6], [1.0, 0.8], default=0.5)
            filtered_df['volatility_score'] = 1 / (filtered_df['atr'] + 1)
            filtered_df['rs_rating_score'] = filtered_df['rs_rating'] / 100
            filtered_df['volume_score'] = np.clip(filtered_df['volume_ratio'] - 1, 0, 1)

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
        cursor.execute("DELETE FROM screener_results")

        for _, row in df.iterrows():
            cursor.execute(
                """INSERT INTO screener_results (
                    symbol, current_price, crossover_date, adx, rsi, rpi, volume_ratio, 
                    support, resistance, dist_ema11_pct, dist_ema21_pct, 
                    fifty_two_week_low, fifty_two_week_high, rank, is_filtered
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row['symbol'], row['current_price'], row['crossover_date'],
                    row['adx'], row['rsi'], row['rpi'], row['volume_ratio'],
                    row['support'], row['resistance'], row['dist_ema11_pct'], row['dist_ema21_pct'],
                    row['fifty_two_week_low'], row['fifty_two_week_high'],
                    row.get('rank'), row['is_filtered']
                )
            )
        db.commit()
        db.close()
        print(f"Screener process finished. Saved {len(df)} stocks to the database.")

    except Exception as e:
        print(f"A critical error occurred during the screener process: {e}")

if __name__ == '__main__':
    run_screener_process()
