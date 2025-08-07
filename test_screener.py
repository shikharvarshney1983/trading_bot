# screener.py
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from datetime import date, timedelta
import sqlite3
import numpy as np
import os
import argparse

# --- Configuration ---
DB_PATH = os.path.join('instance', 'trading.db')

# Configuration for different timeframes
# Defines parameters for indicators and data fetching for each frequency.
TIME_CONFIGS = {
    'daily': {
        'interval': '1d',
        'history_days': 365 * 2,
        'rpi_period': 50,
        'year_lookback': 252,
        'quarter_lookback': 63,
        'crossover_unit_name': 'Days'
    },
    'weekly': {
        'interval': '1wk',
        'history_days': 365 * 5,
        'rpi_period': 20,
        'year_lookback': 52,
        'quarter_lookback': 13,
        'crossover_unit_name': 'Weeks'
    },
    'monthly': {
        'interval': '1mo',
        'history_days': 365 * 10,
        'rpi_period': 12,
        'year_lookback': 12,
        'quarter_lookback': 3,
        'crossover_unit_name': 'Months'
    }
}

def get_screener_end_date(frequency):
    """
    Determines the correct end date for the analysis based on the frequency.
    - Daily: Uses the current day.
    - Weekly: Uses the last completed Friday. This ensures we don't use partial weekly data.
              For example, running this on a Wednesday will use the previous Friday's data.
    - Monthly: Uses the last day of the previous month.
    """
    today = date.today()
    if frequency == 'daily':
        return today
    elif frequency == 'weekly':
        # Go back to the last Monday and then subtract 3 more days to get the previous Friday.
        last_monday = today - timedelta(days=today.weekday())
        last_friday = last_monday - timedelta(days=3)
        return last_friday
    elif frequency == 'monthly':
        # Get the first day of the current month, then subtract one day to get the last day of the previous month.
        first_day_of_current_month = today.replace(day=1)
        last_day_of_previous_month = first_day_of_current_month - timedelta(days=1)
        return last_day_of_previous_month
    else:
        raise ValueError("Invalid frequency provided. Choose from 'daily', 'weekly', 'monthly'.")

def get_nse_stocks_from_db():
    """
    Fetches the list of stock symbols from the master_stocks table in the database.
    """
    print("Fetching stock list from the database...")
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file not found at '{DB_PATH}'")
        return pd.DataFrame()
        
    db = sqlite3.connect(DB_PATH)
    try:
        stocks_df = pd.read_sql_query("SELECT symbol, name, industry, sector FROM master_stocks", db)
        print(f"Found {len(stocks_df)} stocks in the master list.")
        return stocks_df
    except Exception as e:
        print(f"Error fetching stocks from database: {e}")
        return pd.DataFrame()
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
    data['Volume_MA10'] = data['Volume'].rolling(window=10, min_periods=1).mean().squeeze()
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

def get_crossover_periods(data):
    """Calculates how many periods ago the EMA 11/21 crossover occurred."""
    cross = (data['EMA_11'] > data['EMA_21']).astype(int)
    if cross.iloc[-1] == 0: return 999  # Using 999 to signify no current bullish crossover
    
    # A change from False to True is a diff of 1 (a bullish crossover)
    crossover_points = cross.diff()
    last_crossover_start_index = crossover_points[crossover_points == 1].index.max()
    
    if pd.isna(last_crossover_start_index):
        # If a crossover existed for the whole duration
        return 0 
        
    # Calculate periods since the last crossover event
    periods_since_crossover = len(data) - 1 - data.index.get_loc(last_crossover_start_index)
    return periods_since_crossover


def run_screener(frequency):
    """Main function to run the screening process and save to Excel."""
    print(f"Starting {frequency} screener process...")
    
    # Get configuration for the selected frequency
    config = TIME_CONFIGS[frequency]
    interval = config['interval']
    end_date = get_screener_end_date(frequency)
    start_date = end_date - timedelta(days=config['history_days'])
    
    print(f"Data will be screened for the period ending on: {end_date.strftime('%Y-%m-%d')}")

    master_stocks_df = get_nse_stocks_from_db()
    if master_stocks_df.empty:
        print("No stocks to process. Exiting.")
        return

    nifty_data = yf.download('^NSEI', start=start_date, end=end_date, interval=interval)
    if isinstance(nifty_data.columns, pd.MultiIndex):
        nifty_data.columns = nifty_data.columns.droplevel(1)

    results_list = []
    total_stocks = len(master_stocks_df)

    for i, master_stock in master_stocks_df.iterrows():
        symbol = master_stock['symbol']
        print(f"Processing {i+1}/{total_stocks}: {symbol}")
        try:
            data = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)

            if data.empty or len(data) < 50:
                print(f"  - Skipping {symbol}: Insufficient data.")
                continue

            data = calculate_indicators(data)
            
            # Relative Performance Index (RPI) calculation
            stock_ret = data['Close'].pct_change().rolling(config['rpi_period']).sum()
            nifty_ret_aligned = nifty_data['Close'].pct_change().rolling(config['rpi_period']).sum()
            data['RPI'] = stock_ret / nifty_ret_aligned

            latest = data.iloc[-1]
            support, resistance = find_pivots(data)
            
            # Stage 2 checks
            is_stage_2_short = latest['EMA_40'] > data['EMA_40'].iloc[-5]
            is_stage_2_long = latest['EMA_40'] > data['EMA_40'].iloc[-20]
            
            crossover_col_name = f'Crossover {config["crossover_unit_name"]} Ago'

            stock_data_dict = {
                'Symbol': symbol,
                'Name': master_stock['name'],
                'Industry': master_stock['industry'],
                'Sector': master_stock['sector'],
                'Current Price': latest['Close'],
                'EMA_11': latest.get('EMA_11', 0),
                'EMA_21': latest.get('EMA_21', 0),
                'EMA_40': latest.get('EMA_40', 0),
                'ADX': latest.get('ADX_14', 0),
                'RSI': latest.get('RSI_14', 0),
                'Volume Ratio': latest['Volume'] / latest['Volume_MA10'] if latest.get('Volume_MA10', 0) > 0 else 0,
                'RPI': latest.get('RPI', 0),
                'Support': support,
                'Resistance': resistance,
                '1Y High': data['High'].rolling(config['year_lookback']).max().iloc[-1],
                '1Y Low': data['Low'].rolling(config['year_lookback']).min().iloc[-1],
                'Dist from EMA11 %': ((latest['Close'] - latest.get('EMA_11', 0)) / latest.get('EMA_11', 0)) * 100 if latest.get('EMA_11', 0) > 0 else 0,
                'Dist from EMA21 %': ((latest['Close'] - latest.get('EMA_21', 0)) / latest.get('EMA_21', 0)) * 100 if latest.get('EMA_21', 0) > 0 else 0,
                'Dist from Support %': ((latest['Close'] - support) / support) * 100 if support > 0 else 0,
                'Is in Stage 2 (Short)': is_stage_2_short,
                'Is in Stage 2 (Long)': is_stage_2_long,
                'ATR': latest.get('ATRr_14', 0),
                '3M Pct Change': (data['Close'].iloc[-1] / data['Close'].iloc[-config['quarter_lookback']] - 1) * 100 if len(data) > config['quarter_lookback'] else 0,
                crossover_col_name: get_crossover_periods(data)
            }
            results_list.append(stock_data_dict)

        except Exception as e:
            print(f"  - Could not process {symbol}: {e}")
            continue

    if not results_list:
        print("No stocks could be processed to generate a report.")
        return
        
    df = pd.DataFrame(results_list)

    # Calculate final rank score
    df['crossover_score'] = np.select([df[crossover_col_name] <= 3, df[crossover_col_name] <= 6], [1.0, 0.8], default=0.5)
    df['pivot_dist_score'] = 1 - (df['Dist from EMA11 %'] / 10)
    df['volatility_score'] = 1 / (df['ATR'] + 1)
    df['rs_rating_score'] = (df['3M Pct Change'].rank(pct=True) * 100) / 100
    df['volume_score'] = np.clip(df['Volume Ratio'] - 1, 0, 1)

    weights = {'crossover': 0.25, 'pivot': 0.25, 'volatility': 0.1, 'rs': 0.3, 'volume': 0.1}
    
    df['Final Rank Score'] = (
        df['crossover_score'] * weights['crossover'] +
        df['pivot_dist_score'] * weights['pivot'] +
        df['volatility_score'] * weights['volatility'] +
        df['rs_rating_score'] * weights['rs'] +
        df['volume_score'] * weights['volume']
    )
    
    df = df.sort_values('Final Rank Score', ascending=False).reset_index(drop=True)

    # Save to Excel with a frequency-specific name
    output_file = f'screener_{frequency}_output.xlsx'
    print(f"\nSaving detailed analysis to '{output_file}'...")
    df.to_excel(output_file, index=False)
    print("Screener process finished successfully.")

if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Run a stock screener for different timeframes.")
    parser.add_argument(
        '-f', '--frequency', 
        type=str, 
        choices=['daily', 'weekly', 'monthly'], 
        default='weekly', 
        help="The screening frequency. Can be 'daily', 'weekly', or 'monthly'. Defaults to 'weekly'."
    )
    args = parser.parse_args()
    
    # Run the screener with the specified frequency
    run_screener(args.frequency)
