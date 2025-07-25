-- migration_v3.sql
-- This script adds the backtest_results table to the database.

-- Step 1: Create the new backtest_results table if it doesn't already exist.
CREATE TABLE IF NOT EXISTS backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    params_json TEXT NOT NULL,
    results_json TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id)
);

-- Step 2: Update the database version to mark this migration as complete.
PRAGMA user_version = 3;
