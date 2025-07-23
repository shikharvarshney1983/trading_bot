# migrate_db.py
import sqlite3
import os

# --- Configuration ---
# Ensure this path points to your actual database file
DB_PATH = os.path.join('instance', 'trading.db')
MIGRATION_SCRIPT_PATH = 'migration.sql'

def migrate():
    """
    Applies the migration script to the database.
    """
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file not found at '{DB_PATH}'. Please run the app first to create it.")
        return

    if not os.path.exists(MIGRATION_SCRIPT_PATH):
        print(f"Error: Migration script '{MIGRATION_SCRIPT_PATH}' not found.")
        return

    print(f"Connecting to database at '{DB_PATH}'...")
    db = sqlite3.connect(DB_PATH)
    cursor = db.cursor()

    try:
        # Check current version to prevent re-running the migration
        cursor.execute("PRAGMA user_version;")
        version = cursor.fetchone()[0]
        print(f"Current database version: {version}")

        if version >= 2:
            print("Database is already up to date. No migration needed.")
            return

        print("Applying migration script...")
        with open(MIGRATION_SCRIPT_PATH, 'r') as f:
            sql_script = f.read()
            cursor.executescript(sql_script)

        db.commit()
        print("Migration successful! Database is now at version 2.")

    except Exception as e:
        db.rollback()
        print(f"An error occurred during migration: {e}")
        print("The migration has been rolled back.")
    finally:
        db.close()
        print("Database connection closed.")

if __name__ == '__main__':
    migrate()
