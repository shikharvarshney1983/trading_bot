-- schema.sql

DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS portfolios;
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS master_stocks;
DROP TABLE IF EXISTS app_state;
DROP TABLE IF EXISTS screener_results;
DROP TABLE IF EXISTS backtest_results;

-- Create the necessary tables for the trading bot application

CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    telegram_chat_id TEXT,

    -- Balance
    cash_balance REAL NOT NULL DEFAULT 100000.0,
    total_brokerage REAL NOT NULL DEFAULT 0.0,

    -- Strategy Settings
    execution_interval TEXT NOT NULL DEFAULT '1wk',
    stock_list TEXT,
    base_capital REAL NOT NULL DEFAULT 1000000.0,
    brokerage_per_trade REAL NOT NULL DEFAULT 25.0,
    max_open_positions INTEGER NOT NULL DEFAULT 15,
    tranche_sizes TEXT,
    auto_run_enabled BOOLEAN NOT NULL DEFAULT 0,

    -- New columns for decoupled buy workflow
    daily_watchlist TEXT,         -- Stores the comma-separated list of potential buys generated at 4 PM
    next_day_buy_list TEXT      -- Stores the user's final, manually entered list of stocks to buy
);

CREATE TABLE portfolios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    avg_price REAL NOT NULL,
    total_investment REAL NOT NULL,
    tranche_level INTEGER NOT NULL,
    entry_date TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id),
    UNIQUE (user_id, ticker)
);

CREATE TABLE transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    action TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    price REAL NOT NULL,
    value REAL NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id)
);

CREATE TABLE master_stocks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT UNIQUE NOT NULL,
    name TEXT,
    industry TEXT,
    sector TEXT,
    date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE screener_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    frequency TEXT NOT NULL,
    current_price REAL,
    crossover_date TEXT,
    adx REAL,
    rsi REAL,
    rpi REAL,
    volume_ratio REAL,
    support REAL,
    resistance REAL,
    dist_ema11_pct REAL,
    dist_ema21_pct REAL,
    fifty_two_week_low REAL,
    fifty_two_week_high REAL,
    rank INTEGER,
    is_filtered BOOLEAN NOT NULL DEFAULT 0,
    run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    params_json TEXT NOT NULL,
    results_json TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id)
);

CREATE TABLE app_state (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Initialize status for each frequency
INSERT INTO app_state (key, value) VALUES ('screener_status_daily', 'idle');
INSERT INTO app_state (key, value) VALUES ('screener_status_weekly', 'idle');
INSERT INTO app_state (key, value) VALUES ('screener_status_monthly', 'idle');
