-- migration.sql
-- This script migrates the database schema to support multi-frequency screeners
-- without losing existing data.

-- Step 1: Rename the existing screener_results table
ALTER TABLE screener_results RENAME TO screener_results_old;

-- Step 2: Create the new screener_results table with the added 'frequency' column
CREATE TABLE screener_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    frequency TEXT NOT NULL, -- New column
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

-- Step 3: Copy data from the old table to the new one, setting 'weekly' as the default frequency for old runs.
-- Note: We are mapping the old columns to the new table structure.
INSERT INTO screener_results (
    id, symbol, frequency, current_price, crossover_date, adx, rsi, rpi, volume_ratio,
    support, resistance, dist_ema11_pct, dist_ema21_pct, fifty_two_week_low,
    fifty_two_week_high, rank, is_filtered, run_date
)
SELECT
    id, symbol, 'weekly', current_price, crossover_date, adx, rsi, rpi, volume_ratio,
    support, resistance, dist_ema11_pct, dist_ema21_pct, fifty_two_week_low,
    fifty_two_week_high, rank, is_filtered, run_date
FROM screener_results_old;

-- Step 4: Drop the old table
DROP TABLE screener_results_old;

-- Step 5: Update the app_state table for frequency-specific statuses
-- Delete the old generic status key
DELETE FROM app_state WHERE key = 'screener_status';

-- Insert the new status keys, ignoring if they already exist
INSERT OR IGNORE INTO app_state (key, value) VALUES ('screener_status_daily', 'idle');
INSERT OR IGNORE INTO app_state (key, value) VALUES ('screener_status_weekly', 'idle');
INSERT OR IGNORE INTO app_state (key, value) VALUES ('screener_status_monthly', 'idle');

-- Optional: Mark the database as having run this migration
PRAGMA user_version = 2;
