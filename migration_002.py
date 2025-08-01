# migration_001.py
import sqlite3
import os
import logging

# --- Configuration ---
# Ensure this path points to your database file
DB_PATH = os.path.join('instance', 'trading.db')
TABLE_NAME = 'screener_results'
COLUMNS_TO_ADD = {
    'ema_11': 'REAL',
    'ema_21': 'REAL',
    'ema_40': 'REAL',
    'ema_40_prev': 'REAL',
    'prev_close': 'REAL'
}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def migrate():
    """
    Adds new columns to the screener_results table if they don't already exist.
    This allows for schema updates without dropping the table and losing data.
    """
    if not os.path.exists(DB_PATH):
        logging.error(f"Database file not found at '{DB_PATH}'. Please ensure the path is correct.")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Get the list of existing columns in the table
        cursor.execute(f"PRAGMA table_info({TABLE_NAME});")
        existing_columns = [row[1] for row in cursor.fetchall()]
        logging.info(f"Existing columns in '{TABLE_NAME}': {existing_columns}")

        # Loop through the columns we want to add
        for column_name, column_type in COLUMNS_TO_ADD.items():
            if column_name not in existing_columns:
                logging.info(f"Column '{column_name}' not found. Adding it...")
                try:
                    # Use ALTER TABLE to add the new column
                    alter_query = f"ALTER TABLE {TABLE_NAME} ADD COLUMN {column_name} {column_type};"
                    cursor.execute(alter_query)
                    logging.info(f"Successfully added column '{column_name}' to '{TABLE_NAME}'.")
                except sqlite3.OperationalError as e:
                    logging.error(f"Failed to add column '{column_name}': {e}")
            else:
                logging.info(f"Column '{column_name}' already exists. Skipping.")

        # Commit changes and close the connection
        conn.commit()
        conn.close()
        logging.info("Migration check complete.")

    except sqlite3.Error as e:
        logging.error(f"An error occurred while connecting to the database: {e}")

if __name__ == '__main__':
    # Run the migration function when the script is executed
    migrate()
