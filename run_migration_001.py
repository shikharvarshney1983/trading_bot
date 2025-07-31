# run_migration_001.py
#
# This script adds the 'daily_watchlist' and 'next_day_buy_list' columns
# to the 'users' table in the production database.
# It is designed to be run once and is safe to run multiple times.
#
import sqlite3
import os

DB_PATH = os.path.join('instance', 'trading.db')

def migrate():
    """
    Applies the database migration to add new columns to the users table.
    """
    print("Starting database migration...")

    if not os.path.exists(DB_PATH):
        print(f"Error: Database file not found at {DB_PATH}")
        print("Please ensure the database exists before running the migration.")
        return

    try:
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # --- Check for existing columns ---
        cursor.execute("PRAGMA table_info(users)")
        columns = [info[1] for info in cursor.fetchall()]

        # --- Add 'daily_watchlist' column if it doesn't exist ---
        if 'daily_watchlist' not in columns:
            print("Adding 'daily_watchlist' column to 'users' table...")
            cursor.execute("ALTER TABLE users ADD COLUMN daily_watchlist TEXT")
            print("'daily_watchlist' column added successfully.")
        else:
            print("'daily_watchlist' column already exists. Skipping.")

        # --- Add 'next_day_buy_list' column if it doesn't exist ---
        if 'next_day_buy_list' not in columns:
            print("Adding 'next_day_buy_list' column to 'users' table...")
            cursor.execute("ALTER TABLE users ADD COLUMN next_day_buy_list TEXT")
            print("'next_day_buy_list' column added successfully.")
        else:
            print("'next_day_buy_list' column already exists. Skipping.")

        # Commit the changes to the database
        conn.commit()
        print("\nMigration completed successfully!")

    except sqlite3.Error as e:
        print(f"\nAn error occurred during migration: {e}")
        if 'conn' in locals() and conn:
            conn.rollback()
    finally:
        # Ensure the connection is closed
        if 'conn' in locals() and conn:
            conn.close()
            print("Database connection closed.")

if __name__ == '__main__':
    migrate()
