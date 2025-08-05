import pandas as pd
import yfinance as yf
import pandas_ta as ta
import sqlite3
from sqlalchemy import create_engine
import schedule
import time
from datetime import datetime, date
import pytz
import os
import numpy as np

# --- CONFIGURATION ---
DB_FILE = "financial_data.db"
STOCK_LIST_FILE = "stocks.txt" # A text file with one stock symbol per line (e.g., RELIANCE.NS)
NIFTY_TICKER = "^NSEI" # NIFTY 50 ticker for relative strength calculation
SCHEDULE_TIME = "15:40" # IST time to run the daily update
INDICATOR_LOOKBACK_BUFFER = 300 # Days/weeks of extra data to load for accurate indicator calculation

# --- DATABASE SETUP ---
# Using SQLAlchemy for easier interaction between pandas DataFrame and SQL
db_engine = create_engine(f'sqlite:///{DB_FILE}')

def setup_database():
    """
    Initializes the database and creates tables if they don't exist.
    """
    print("Setting up database...")
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        # Stock Master Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS master_stock (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE NOT NULL,
                name TEXT,
                sector TEXT,
                industry TEXT
            )
        ''')
        # Daily Price Data Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_price (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_id INTEGER,
                date DATE,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                FOREIGN KEY (stock_id) REFERENCES master_stock (id),
                UNIQUE (stock_id, date)
            )
        ''')
        # Weekly Price Data Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weekly_price (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_id INTEGER,
                date DATE,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                FOREIGN KEY (stock_id) REFERENCES master_stock (id),
                UNIQUE (stock_id, date)
            )
        ''')
        # Indicators tables will be created dynamically when data is first inserted.
    print("Database setup complete.")


# --- DATA FETCHING & PROCESSING ---

def clear_all_transactional_data():
    """
    Clears all price and indicator data from the database.
    This is intended for a full weekend refresh to account for corporate actions.
    """
    print("--- WEEKEND REFRESH: Clearing all price and indicator data... ---")
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM daily_price")
            cursor.execute("DELETE FROM weekly_price")
            # Indicator tables are fully replaced, so dropping them is cleaner.
            cursor.execute("DROP TABLE IF EXISTS daily_indicators")
            cursor.execute("DROP TABLE IF EXISTS weekly_indicators")
            conn.commit()
            print("--- All transactional data cleared successfully. ---")
        except Exception as e:
            print(f"Error while clearing data: {e}")

def update_stock_master():
    """
    Reads the stock list file and updates the master_stock table with any new symbols.
    Fetches metadata for new stocks from yfinance.
    """
    print("\nUpdating stock master list...")
    if not os.path.exists(STOCK_LIST_FILE):
        print(f"Error: {STOCK_LIST_FILE} not found. Please create it with stock symbols.")
        # Create a dummy file for demonstration
        with open(STOCK_LIST_FILE, "w") as f:
            f.write("RELIANCE.NS\n")
            f.write("TCS.NS\n")
            f.write("HDFCBANK.NS\n")
        print(f"Created a sample {STOCK_LIST_FILE}.")

    with open(STOCK_LIST_FILE, 'r') as f:
        symbols = [line.strip() for line in f if line.strip()]

    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        for symbol in symbols:
            cursor.execute("SELECT id FROM master_stock WHERE symbol = ?", (symbol,))
            if cursor.fetchone() is None:
                print(f"New stock found: {symbol}. Fetching info...")
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    name = info.get('longName', symbol)
                    sector = info.get('sector', 'N/A')
                    industry = info.get('industry', 'N/A')
                    cursor.execute(
                        "INSERT INTO master_stock (symbol, name, sector, industry) VALUES (?, ?, ?, ?)",
                        (symbol, name, sector, industry)
                    )
                    print(f"Added {symbol} to master list.")
                except Exception as e:
                    print(f"Could not fetch info for {symbol}: {e}")
    print("Stock master update complete.")

def get_last_date(stock_id, timeframe, table_prefix=""):
    """Gets the last entry date for a stock from a given table."""
    table_name = f"{timeframe}_{table_prefix}price" if table_prefix == "" else f"{timeframe}_indicators"
    try:
        with sqlite3.connect(DB_FILE) as conn:
            # Check if table exists first
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            if cursor.fetchone() is None:
                return None # Table doesn't exist

            # Table exists, get max date
            cursor.execute(f"SELECT MAX(date) FROM {table_name} WHERE stock_id = ?", (stock_id,))
            result = cursor.fetchone()[0]
            if result:
                return pd.to_datetime(result).date()
    except Exception:
        return None
    return None

def update_price_data():
    """
    Fetches and updates daily and weekly price data for all stocks in the master list.
    Performs incremental updates to only fetch new data.
    """
    print("\nUpdating price data...")
    end_date_str = (datetime.today() + pd.Timedelta(days=1)).strftime('%Y-%m-%d') # Fetch up to tomorrow to include today's data
    with sqlite3.connect(DB_FILE) as conn:
        master_list = pd.read_sql("SELECT id, symbol FROM master_stock", conn)

    for _, row in master_list.iterrows():
        stock_id, symbol = row['id'], row['symbol']
        print(f"-- Processing {symbol} --")
        for timeframe in ['daily', 'weekly']:
            interval = '1d' if timeframe == 'daily' else '1wk'
            last_date = get_last_date(stock_id, timeframe)
            start_date = (last_date + pd.Timedelta(days=1)) if last_date else "1990-01-01"

            if last_date and last_date >= date.today():
                 print(f"{symbol} {timeframe} data is already up-to-date.")
                 continue

            try:
                # Use auto_adjust=True to handle splits/dividends automatically
                df = yf.download(
                    symbol, 
                    start=start_date, 
                    end=end_date_str, 
                    interval=interval, 
                    progress=False, 
                    auto_adjust=True
                )
                if not df.empty:
                    # yfinance can return a multi-index column. This checks for
                    # that case and flattens the columns to a simple, single-level index.
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                    df_to_save = pd.DataFrame({
                        'stock_id': stock_id,
                        'date': df.index.date,
                        'open': df['Open'].values,
                        'high': df['High'].values,
                        'low': df['Low'].values,
                        'close': df['Close'].values,
                        'volume': df['Volume'].values
                    })
                    
                    # Save to DB
                    df_to_save.to_sql(
                        f'{timeframe}_price',
                        db_engine,
                        if_exists='append',
                        index=False
                    )
                    print(f"Updated {symbol} with {len(df_to_save)} new {timeframe} records.")
            except sqlite3.IntegrityError:
                # This is a more robust way to catch the unique constraint violation
                print(f"Data for {symbol} on some dates already exists. Skipping.")
            except Exception as e:
                print(f"Could not process and save price data for {symbol}: {e}")
    print("Price data update complete.")

def calculate_and_store_indicators():
    """
    Calculates all required indicators for both daily and weekly timeframes
    and stores them in the database.
    """
    print("\nCalculating indicators...")
    # First, get Nifty data for relative strength calculation
    nifty_daily = yf.download(NIFTY_TICKER, period="max", interval="1d", auto_adjust=True)
    nifty_weekly = yf.download(NIFTY_TICKER, period="max", interval="1wk", auto_adjust=True)

    # Flatten the Nifty dataframe columns if they are a MultiIndex
    if isinstance(nifty_daily.columns, pd.MultiIndex):
        nifty_daily.columns = nifty_daily.columns.get_level_values(0)
    if isinstance(nifty_weekly.columns, pd.MultiIndex):
        nifty_weekly.columns = nifty_weekly.columns.get_level_values(0)

    nifty_map = {'daily': nifty_daily, 'weekly': nifty_weekly}

    with sqlite3.connect(DB_FILE) as conn:
        master_list = pd.read_sql("SELECT id, symbol FROM master_stock", conn)

    for timeframe in ['daily', 'weekly']:
        all_indicators_to_append = []
        print(f"\n--- Processing {timeframe.upper()} Indicators ---")
        for _, row in master_list.iterrows():
            stock_id, symbol = row['id'], row['symbol']
            
            last_indicator_date = get_last_date(stock_id, timeframe, table_prefix="indicators")

            # Load only the necessary slice of data
            # We need enough historical data to calculate the longest lookback indicator
            if last_indicator_date:
                # Load new data plus a buffer for calculation
                start_fetch_date = last_indicator_date - pd.Timedelta(days=INDICATOR_LOOKBACK_BUFFER if timeframe == 'daily' else INDICATOR_LOOKBACK_BUFFER * 7)
                sql_query = f"SELECT * FROM {timeframe}_price WHERE stock_id = {stock_id} AND date >= '{start_fetch_date}'"
            else:
                # First time, load all data
                sql_query = f"SELECT * FROM {timeframe}_price WHERE stock_id = {stock_id}"
            
            price_df = pd.read_sql(sql_query, db_engine, parse_dates=['date'], index_col='date')
            price_df = price_df[~price_df.index.duplicated(keep='first')]

            if price_df.empty or len(price_df) < 50:
                continue

            # 1. Standard Indicators using pandas_ta
            custom_strategy = ta.Strategy(
                name="All_Indicators",
                description="A collection of common technical indicators.",
                ta=[
                    {"kind": "adx", "length": 14}, {"kind": "rsi", "length": 14},
                    {"kind": "atr", "length": 14}, {"kind": "ema", "length": 11},
                    {"kind": "ema", "length": 21}, {"kind": "ema", "length": 40},
                    {"kind": "ema", "length": 150}, {"kind": "sma", "length": 11},
                    {"kind": "sma", "length": 21}, {"kind": "sma", "length": 40},
                    {"kind": "sma", "length": 150}, {"kind": "psar"},
                    {"kind": "macd"}, {"kind": "bbands", "length": 20},
                    {"kind": "donchian", "lower_length": 20, "upper_length": 20},
                    {"kind": "willr", "length": 14}
                ]
            )
            price_df.ta.strategy(custom_strategy)

            # 2. Custom Indicators
            price_df['vstop'] = price_df['low'] - price_df['ATRr_14'] * 2
            
            nifty_data = nifty_map[timeframe]
            nifty_close_df = nifty_data[['Close']].rename(columns={'Close': 'nifty_close'})
            merged = price_df.join(nifty_close_df)
            merged['nifty_close'].fillna(method='ffill', inplace=True)
            
            lookback = 50 if timeframe == 'daily' else 11
            stock_move = merged['close'].pct_change(periods=lookback)
            nifty_move = merged['nifty_close'].pct_change(periods=lookback)
            price_df['rs_nifty'] = ((1 + stock_move) / (1 + nifty_move)) - 1
            
            price_df['volume_sma_10'] = price_df['volume'].rolling(10).mean()
            rolling_window = 252 if timeframe == 'daily' else 52
            price_df['52w_high'] = price_df['high'].rolling(rolling_window).max()
            price_df['52w_low'] = price_df['low'].rolling(rolling_window).min()
            price_df['support'] = price_df['low'].rolling(10, center=True).min()
            price_df['resistance'] = price_df['high'].rolling(10, center=True).max()
            price_df[['support', 'resistance']] = price_df[['support', 'resistance']].fillna(method='ffill')

            # Clean up and prepare for saving
            indicators_df = price_df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'id', 'stock_id'], errors='ignore').reset_index()
            indicators_df['stock_id'] = stock_id
            indicators_df.columns = [col.lower().replace('_{"length": 20, "std": 2.0}', '') for col in indicators_df.columns]
            
            # Filter for only the new rows that need to be appended
            if last_indicator_date:
                indicators_df = indicators_df[indicators_df['date'].dt.date > last_indicator_date]

            if not indicators_df.empty:
                all_indicators_to_append.append(indicators_df)
                print(f"Calculated {len(indicators_df)} new {timeframe} indicator rows for {symbol}.")

        # After the loop, concatenate and save all data for the timeframe at once
        if all_indicators_to_append:
            final_df = pd.concat(all_indicators_to_append, ignore_index=True)
            final_df.to_sql(f'{timeframe}_indicators', db_engine, if_exists='append', index=False)
            print(f"\nAppended {len(final_df)} total new {timeframe} indicator rows to the database.")

    print("\nIndicator calculation complete.")


# --- SCHEDULING ---

def run_pipeline():
    """
    The main job function that runs the entire data pipeline.
    On weekends, it clears all data to refresh for corporate actions.
    """
    # Check if it's the weekend (Saturday=5, Sunday=6)
    # We run the full refresh on Saturday.
    if datetime.today().weekday() == 5: # 5 is Saturday
        clear_all_transactional_data()

    print(f"\n--- Starting Daily Data Pipeline at {datetime.now()} ---")
    setup_database()
    update_stock_master()
    update_price_data()
    calculate_and_store_indicators()
    print(f"--- Pipeline Run Finished at {datetime.now()} ---\n")

def main():
    """
    Main function to run the script either once or on a schedule.
    """
    # For testing, run it once immediately
    run_pipeline()

    # Set up the scheduler
    print(f"Scheduling job to run every day at {SCHEDULE_TIME} IST.")
    # Note: The 'schedule' library uses the server's local time.
    # Ensure the server's timezone is set to IST for this to work as expected.
    schedule.every().day.at(SCHEDULE_TIME).do(run_pipeline)
    
    while True:
        schedule.run_pending()
        time.sleep(30) # Check every 30 seconds


if __name__ == '__main__':
    main()
