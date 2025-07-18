-- schema.sql

DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS portfolios;
DROP TABLE IF EXISTS transactions;

CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    telegram_chat_id TEXT, -- Added for Telegram notifications

    -- Balance
    cash_balance REAL NOT NULL DEFAULT 100000.0,
    total_brokerage REAL NOT NULL DEFAULT 0.0,

    -- Strategy Settings
    execution_interval TEXT NOT NULL DEFAULT '1wk',
    stock_list TEXT,
    base_capital REAL NOT NULL DEFAULT 1000000.0,
    brokerage_per_trade REAL NOT NULL DEFAULT 25.0,
    max_open_positions INTEGER NOT NULL DEFAULT 15,
    tranche_sizes TEXT
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
