# database.py

import sqlite3
import click
from flask.cli import with_appcontext
from werkzeug.security import generate_password_hash

DB_NAME = 'trading.db'

def get_db():
    """Connects to the application's configured database."""
    db = sqlite3.connect(
        f'instance/{DB_NAME}',
        detect_types=sqlite3.PARSE_DECLTYPES
    )
    db.row_factory = sqlite3.Row
    return db

def close_db(e=None):
    """Closes the database connection."""
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    """Initializes the database by creating tables."""
    db = get_db()
    with open('schema.sql', 'r') as f:
        db.executescript(f.read())
    db.close()
    print("Database has been initialized.")
    
def create_admin():
    """Creates the default admin user."""
    db = get_db()
    try:
        # The default admin password is 'admin'
        password_hash = generate_password_hash('admin')
        db.execute(
            'INSERT INTO users (username, password_hash, role, cash_balance) VALUES (?, ?, ?, ?)',
            ('admin', password_hash, 'admin', 1000000)
        )
        db.commit()
        print("Admin user 'admin' created with password 'admin'.")
    except db.IntegrityError:
        print("Admin user already exists.")
    finally:
        db.close()

@click.command('init-db')
@with_appcontext
def init_db_command():
    """CLI command to initialize the database."""
    init_db()
    create_admin()

def init_app(app):
    """Registers database functions with the Flask app."""
    app.cli.add_command(init_db_command)
