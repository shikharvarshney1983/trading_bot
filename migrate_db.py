# migrate_db.py
import sqlite3
import os

# --- Configuration ---
DB_PATH = os.path.join('instance', 'trading.db')
MIGRATIONS_DIR = 'migrations' # A directory to hold migration scripts

def get_db_version(cursor):
    """Gets the current user_version from the database."""
    cursor.execute("PRAGMA user_version;")
    return cursor.fetchone()[0]

def apply_migration(db, cursor, version):
    """Applies a single migration script."""
    script_path = os.path.join(MIGRATIONS_DIR, f'v{version}_migration.sql')
    if not os.path.exists(script_path):
        print(f"Warning: Migration script not found: {script_path}")
        return False
    
    print(f"Applying migration: {script_path}...")
    with open(script_path, 'r') as f:
        sql_script = f.read()
        cursor.executescript(sql_script)
    
    # Verify that the migration script updated the version correctly
    if get_db_version(cursor) != version:
        raise Exception(f"Migration script {script_path} did not update the DB version to {version}!")

    db.commit()
    print(f"Successfully applied version {version} migration.")
    return True

def migrate():
    """
    Checks the database version and applies all necessary migrations sequentially.
    """
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file not found at '{DB_PATH}'. Please run the app first to create it.")
        return

    if not os.path.exists(MIGRATIONS_DIR):
        os.makedirs(MIGRATIONS_DIR)
        print(f"Created migrations directory at '{MIGRATIONS_DIR}'.")
        print("Please place your migration scripts (e.g., v3_migration.sql) there.")
        return

    print(f"Connecting to database at '{DB_PATH}'...")
    db = sqlite3.connect(DB_PATH)
    cursor = db.cursor()

    try:
        current_version = get_db_version(cursor)
        print(f"Current database version: {current_version}")
        
        # This is where you define the latest version of your schema.
        # The script will try to apply migrations from current_version + 1 up to this version.
        latest_version = 3 

        if current_version >= latest_version:
            print("Database is already up to date.")
            return

        for version in range(current_version + 1, latest_version + 1):
            if not apply_migration(db, cursor, version):
                # Stop if a migration script is missing
                break
        
        print("All necessary migrations have been applied.")

    except Exception as e:
        db.rollback()
        print(f"An error occurred during migration: {e}")
        print("The migration has been rolled back.")
    finally:
        db.close()
        print("Database connection closed.")

if __name__ == '__main__':
    # Rename your 'migration.sql' to 'v2_migration.sql' and the new one to 'v3_migration.sql'
    # inside a 'migrations' folder for this script to work.
    migrate()
