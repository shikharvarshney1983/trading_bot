# app.py

import os
import json
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import pytz
import requests
from dotenv import load_dotenv
from flask import (
    Flask, jsonify, request, render_template, redirect, url_for, flash, send_file
)
from werkzeug.security import check_password_hash, generate_password_hash
from flask_login import (
    LoginManager, UserMixin, login_user, login_required, logout_user, current_user
)
from functools import wraps
from apscheduler.schedulers.background import BackgroundScheduler
import io, logging
import threading
from collections import deque


# --- Centralized Logging Configuration ---
# This setup is done once to avoid duplicate handlers and encoding errors.
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler = logging.FileHandler('trading_bot.log', encoding='utf-8') # Specify UTF-8 encoding
log_handler.setFormatter(log_formatter)

# Get the root logger
logger = logging.getLogger()
if not logger.handlers: # Add handler only if no handlers are configured
    logger.addHandler(log_handler)
    logger.setLevel(logging.INFO)


# Local imports
import database
from trading_bot import TradingBot
import stock_screener # Import the new screener script
from backtester import Backtester # Import the new backtester class
from strategy import MomentumStrategy # Import the new strategy class
from data_utils import get_data_with_indicators

# --- Load Environment Variables ---
load_dotenv()

# --- App Configuration ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a-very-secret-key-that-should-be-changed'
app.config['DATABASE'] = os.path.join(app.instance_path, database.DB_NAME)
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')


# --- Global Cache for Live Prices ---
live_prices_cache = {}
cache_expiry = timedelta(minutes=1)
last_cache_update = datetime.min

# --- Default Constants for New Users ---
NEW_USER_BASE_CAPITAL = 1000000.0 # Default base capital for new users
NEW_USER_STOCKS = "RELIANCE.NS,TCS.NS,HDFCBANK.NS"
NEW_USER_TRANCHES = {
    "1": 0.01, "2": 0.01, "3": 0.01, "4": 0.01, "5": 0.01
}

try:
    os.makedirs(app.instance_path)
except OSError:
    pass

database.init_app(app)

# --- Login Manager Setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username, role):
        self.id = id
        self.username = username
        self.role = role

@login_manager.user_loader
def load_user(user_id):
    db = database.get_db()
    user_row = db.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    db.close()
    if user_row:
        return User(id=user_row['id'], username=user_row['username'], role=user_row['role'])
    return None

# --- Helper Functions & Decorators ---
def send_telegram_message(chat_id, message):
    """Sends a message to a given Telegram chat ID."""
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        logging.info("Telegram token or chat ID is missing. Skipping notification.")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': chat_id, 'text': message, 'parse_mode': 'Markdown'}
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            logging.warning(f"Error sending Telegram message: {response.text}")
    except Exception as e:
        logging.error(f"Exception while sending Telegram message: {e}")

def is_market_open():
    """Checks if the Indian stock market is currently open."""
    tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(tz)
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close and now.weekday() < 5

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            flash("You do not have permission to access this page.", "danger")
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# --- Authentication Routes ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = database.get_db()
        user_row = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        db.close()
        
        if user_row and check_password_hash(user_row['password_hash'], password):
            user = User(id=user_row['id'], username=user_row['username'], role=user_row['role'])
            login_user(user)
            flash(f"Welcome back, {user.username}!", "success")
            return redirect(url_for('admin_panel') if user.role == 'admin' else url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
            
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

# --- Admin Routes ---
@app.route('/admin')
@login_required
@admin_required
def admin_panel():
    db = database.get_db()
    users = db.execute('SELECT id, username, role, cash_balance FROM users').fetchall()
    db.close()
    return render_template('admin.html', users=users)

@app.route('/admin/create_user', methods=['POST'])
@login_required
@admin_required
def create_user():
    username = request.form['username']
    password = request.form['password']
    role = request.form['role']
    
    if not username or not password:
        flash("Username and password are required.", "danger")
        return redirect(url_for('admin_panel'))
        
    db = database.get_db()
    try:
        cash_balance = NEW_USER_BASE_CAPITAL * 0.4
        db.execute(
            '''INSERT INTO users (username, password_hash, role, base_capital, cash_balance, stock_list, tranche_sizes) 
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (username, generate_password_hash(password), role, 
             NEW_USER_BASE_CAPITAL, cash_balance, NEW_USER_STOCKS, json.dumps(NEW_USER_TRANCHES))
        )
        db.commit()
        flash(f"User '{username}' created successfully.", "success")
    except db.IntegrityError:
        flash(f"Username '{username}' already exists.", "danger")
    finally:
        db.close()
        
    return redirect(url_for('admin_panel'))

@app.route('/admin/update_balance', methods=['POST'])
@login_required
@admin_required
def admin_update_balance():
    user_id = request.form['user_id']
    new_balance = float(request.form['cash_balance'])
    
    db = database.get_db()
    db.execute('UPDATE users SET cash_balance = ? WHERE id = ?', (new_balance, user_id))
    db.commit()
    db.close()
    
    flash("User balance updated.", "success")
    return redirect(url_for('admin_panel'))

@app.route('/admin/change_password', methods=['POST'])
@login_required
@admin_required
def admin_change_password():
    user_id = request.form['user_id']
    new_password = request.form['new_password']

    if not new_password:
        flash("Password cannot be empty.", "danger")
        return redirect(url_for('admin_panel'))

    password_hash = generate_password_hash(new_password)
    db = database.get_db()
    db.execute('UPDATE users SET password_hash = ? WHERE id = ?', (password_hash, user_id))
    db.commit()
    db.close()
    
    flash("User password has been changed.", "success")
    return redirect(url_for('admin_panel'))

@app.route('/admin/delete_user', methods=['POST'])
@login_required
@admin_required
def delete_user():
    user_id_to_delete = request.form['user_id']

    if int(user_id_to_delete) == current_user.id:
        flash("You cannot delete your own account.", "danger")
        return redirect(url_for('admin_panel'))

    db = database.get_db()
    try:
        db.execute('DELETE FROM transactions WHERE user_id = ?', (user_id_to_delete,))
        db.execute('DELETE FROM portfolios WHERE user_id = ?', (user_id_to_delete,))
        db.execute('DELETE FROM users WHERE id = ?', (user_id_to_delete,))
        db.commit()
        flash("User deleted successfully.", "success")
    except Exception as e:
        db.rollback()
        flash(f"Error deleting user: {e}", "danger")
    finally:
        db.close()
        
    return redirect(url_for('admin_panel'))

# --- User Dashboard and API Routes ---
@app.route('/')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/tcpl')
def tcpl():
    return render_template('tcplanalysis.html')

# --- Backtester Routes ---
@app.route('/backtest')
@login_required
def backtest_page():
    db = database.get_db()
    user_settings = db.execute('SELECT * FROM users WHERE id = ?', (current_user.id,)).fetchone()
    db.close()
    return render_template('backtest.html', user_settings=user_settings)

@app.route('/api/run_backtest', methods=['POST'])
@login_required
def run_backtest():
    logging.info(f"--- INITIATING BACKTEST for user '{current_user.username}' ---")
    params = request.json
    try:
        start_date = datetime.strptime(params['start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(params['end_date'], '%Y-%m-%d')
        
        strategy_instance = MomentumStrategy()
        
        backtester = Backtester(
            stock_tickers=params['stock_list'],
            start_date=start_date,
            end_date=end_date,
            interval=params['interval'],
            initial_capital=params['initial_capital'],
            strategy_capital=params['strategy_capital'],
            tranche_sizes_pct=json.loads(params['tranche_sizes']),
            brokerage=params['brokerage'],
            max_positions=params['max_positions'],
            strategy=strategy_instance
        )
        results = backtester.run()

        db = database.get_db()
        db.execute(
            'INSERT INTO backtest_results (user_id, params_json, results_json) VALUES (?, ?, ?)',
            (current_user.id, json.dumps(params), json.dumps(results))
        )
        db.commit()
        db.close()
        
        logging.info(f"--- COMPLETED BACKTEST for user '{current_user.username}' ---")
        return jsonify({'status': 'success', 'results': results})
    except Exception as e:
        logging.error(f"Backtest error for user '{current_user.username}': {e}")
        log_data = backtester.log if 'backtester' in locals() else []
        return jsonify({'status': 'error', 'message': str(e), 'log': log_data}), 500

@app.route('/api/backtest_results')
@login_required
def get_backtest_results():
    db = database.get_db()
    results = db.execute('SELECT * FROM backtest_results WHERE user_id = ? ORDER BY run_date DESC', (current_user.id,)).fetchall()
    db.close()
    return jsonify([dict(row) for row in results])

def get_live_prices_bulk(tickers: list):
    global live_prices_cache, last_cache_update
    now = datetime.now()
    
    if now - last_cache_update > cache_expiry:
        live_prices_cache.clear()
        
    tickers_to_fetch = [t for t in tickers if t not in live_prices_cache]
    
    if not tickers_to_fetch:
        return {t: live_prices_cache[t] for t in tickers if t in live_prices_cache}

    logging.info(f"Fetching live prices for: {tickers_to_fetch}")
    try:
        data = yf.download(tickers=tickers_to_fetch, period='5d', progress=False)
        
        if data.empty:
            logging.error(f"yf.download returned empty dataframe for tickers: {tickers_to_fetch}")
            return {}
        
        if isinstance(data.columns, pd.MultiIndex):
            last_prices = data['Close'].ffill().iloc[-1]
        else:
            close_prices = data['Close'].ffill()
            if close_prices.empty:
                 last_price_value = None
            else:
                 last_price_value = close_prices.iloc[-1]
            last_prices = pd.Series({tickers_to_fetch[0]: last_price_value})

        for ticker, price in last_prices.items():
            if pd.notna(price):
                live_prices_cache[ticker] = price
        last_cache_update = now
    except Exception as e:
        logging.error(f"Error in get_live_prices_bulk: {e}", exc_info=True)
        if now - last_cache_update > cache_expiry:
            live_prices_cache.clear()
            
    return {t: live_prices_cache.get(t) for t in tickers}

def calculate_realized_pnl_fifo(transactions):
    investments = {}
    realized_pnl = 0
    daily_pnl = {}

    for tx in transactions:
        ticker = tx['ticker']
        if ticker not in investments:
            investments[ticker] = deque()

        if tx['action'] in ['BUY', 'ADD']:
            investments[ticker].append({'quantity': tx['quantity'], 'price': tx['price']})
        
        elif tx['action'] == 'SELL':
            sell_quantity = tx['quantity']
            sell_value = tx['value']
            cost_of_sold_shares = 0

            while sell_quantity > 0 and investments[ticker]:
                oldest_lot = investments[ticker][0]
                
                if oldest_lot['quantity'] <= sell_quantity:
                    cost_of_sold_shares += oldest_lot['quantity'] * oldest_lot['price']
                    sell_quantity -= oldest_lot['quantity']
                    investments[ticker].popleft()
                else:
                    cost_of_sold_shares += sell_quantity * oldest_lot['price']
                    oldest_lot['quantity'] -= sell_quantity
                    sell_quantity = 0
            
            pnl = sell_value - cost_of_sold_shares
            realized_pnl += pnl
            
            trade_date = datetime.strptime(tx['date'].split(" ")[0], '%Y-%m-%d').strftime('%Y-%m-%d')
            daily_pnl[trade_date] = daily_pnl.get(trade_date, 0) + pnl

    return realized_pnl, daily_pnl


def get_dashboard_data(user_id):
    db = database.get_db()
    user_data = db.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    portfolio_rows = db.execute('SELECT * FROM portfolios WHERE user_id = ?', (user_id,)).fetchall()
    transaction_rows = db.execute('SELECT * FROM transactions WHERE user_id = ? ORDER BY date ASC', (user_id,)).fetchall()
    db.close()

    portfolio = {row['ticker']: {key: row[key] for key in row.keys()} for row in portfolio_rows}
    transactions = [dict(row) for row in transaction_rows]
    
    portfolio_tickers = list(portfolio.keys())
    live_prices = {}
    if portfolio_tickers:
        live_prices = get_live_prices_bulk(portfolio_tickers)

    unrealized_pnl = 0
    holdings_value = 0
    for ticker, pos in portfolio.items():
        last_price = live_prices.get(ticker, pos['avg_price'])
        if last_price:
            pos['current_price'] = last_price
            pos['market_value'] = pos['quantity'] * last_price
            pos['pnl'] = (last_price - pos['avg_price']) * pos['quantity']
            unrealized_pnl += pos['pnl']
            holdings_value += pos['market_value']
        else:
            pos['current_price'] = pos['avg_price']
            pos['market_value'] = pos['quantity'] * pos['avg_price']
            pos['pnl'] = 0

    portfolio_value = holdings_value + user_data['cash_balance']

    realized_pnl, daily_pnl = calculate_realized_pnl_fifo(transactions)

    pnl_values = [pnl for pnl in daily_pnl.values() if pnl != 0]
    wins = sum(1 for pnl in pnl_values if pnl > 0)
    losses = sum(1 for pnl in pnl_values if pnl < 0)
    total_closed = wins + losses
    win_ratio = (wins / total_closed * 100) if total_closed > 0 else 0
    avg_win = sum(pnl for pnl in pnl_values if pnl > 0) / wins if wins > 0 else 0
    avg_loss = sum(pnl for pnl in pnl_values if pnl < 0) / losses if losses > 0 else 0

    transactions.reverse()

    return {
        'portfolio': portfolio,
        'transactions': transactions,
        'stats': {
            'portfolio_value': portfolio_value,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': realized_pnl,
            'win_ratio': win_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'cash_balance': user_data['cash_balance']
        },
        'calendar_pnl': daily_pnl
    }

@app.route('/api/status')
@login_required
def get_status():
    user_id = current_user.id
    db = database.get_db()
    user_data = db.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    db.close()

    dashboard_data = get_dashboard_data(user_id)
    dashboard_data['settings'] = {key: user_data[key] for key in user_data.keys()}
    
    return jsonify(dashboard_data)

@app.route('/api/dynamic_data')
@login_required
def get_dynamic_data():
    user_id = current_user.id
    dashboard_data = get_dashboard_data(user_id)
    return jsonify(dashboard_data)

@app.route('/api/save_buy_list', methods=['POST'])
@login_required
def save_buy_list():
    data = request.json
    buy_list_str = data.get('buy_list', '')
    raw_tickers = [ticker.strip().upper() for ticker in buy_list_str.split(',') if ticker.strip()]
    
    # FIX: Ensure every ticker has the .NS suffix
    processed_tickers = []
    for ticker in raw_tickers:
        if not ticker.endswith('.NS'):
            processed_tickers.append(ticker + '.NS')
        else:
            processed_tickers.append(ticker)
            
    db = database.get_db()
    try:
        db.execute('UPDATE users SET next_day_buy_list = ? WHERE id = ?', (','.join(processed_tickers), current_user.id))
        db.commit()
    except Exception as e:
        db.rollback()
        return jsonify({'status': 'error', 'message': f'Database error: {e}'}), 500
    finally:
        db.close()
    
    return jsonify({'status': 'success', 'message': 'Buy list saved.'})

def get_live_price(ticker):
    prices = get_live_prices_bulk([ticker])
    return prices.get(ticker)

def execute_sell_and_add_trades(user_id):
    db = database.get_db()
    user = None
    try:
        user = db.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        if not user: return
        logging.info(f"--- Executing SELL/ADD for {user['username']} ---")
        portfolio_rows = db.execute('SELECT * FROM portfolios WHERE user_id = ?', (user_id,)).fetchall()
        portfolio = {row['ticker']: dict(row) for row in portfolio_rows}
        if not portfolio:
            logging.info(f"No open positions for {user['username']}. Skipping.")
            db.close()
            return
        
        strategy = MomentumStrategy()
        all_data = get_data_with_indicators(list(portfolio.keys()), '^NSEI', user['execution_interval'])
        if all_data is None: raise Exception("Failed to fetch market data for sell/add.")
        
        cash_balance = user['cash_balance']
        brokerage = user['brokerage_per_trade']
        base_capital = user['base_capital']
        tranches = json.loads(user['tranche_sizes'])

        sell_signals = strategy.get_sell_signals(portfolio, all_data)
        for signal in sell_signals:
            ticker, position = signal['ticker'], portfolio[signal['ticker']]
            sell_price = get_live_price(ticker)
            if sell_price is None: continue
            sell_value = position['quantity'] * sell_price
            cash_balance += sell_value - brokerage
            db.execute('DELETE FROM portfolios WHERE user_id = ? AND ticker = ?', (user_id, ticker))
            db.execute('INSERT INTO transactions (user_id, date, ticker, action, quantity, price, value) VALUES (?, ?, ?, ?, ?, ?, ?)',
                       (user_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ticker, 'SELL', position['quantity'], sell_price, sell_value))
            logging.info(f"SELL: {user['username']} - {ticker} @ {sell_price}")
            if user['telegram_chat_id']:
                message = f"🔴 SELL Order Executed\n\n*Ticker:* {ticker}\n*Quantity:* {position['quantity']}\n*Price:* ₹{sell_price:.2f}\n*Reason:* {signal['reason']}"
                send_telegram_message(user['telegram_chat_id'], message)

        add_on_signals = strategy.get_add_on_signals(portfolio, all_data)
        for signal in add_on_signals:
            ticker, position = signal['ticker'], portfolio[signal['ticker']]
            next_tranche_level = position['tranche_level'] + 1
            if str(next_tranche_level) not in tranches: continue
            add_price = get_live_price(ticker)
            if add_price is None: continue
            target_investment = base_capital * tranches[str(next_tranche_level)]
            add_quantity = round(target_investment / add_price) if add_price > 0 else 0
            trade_cost = (add_quantity * add_price) + brokerage
            if add_quantity > 0 and cash_balance >= trade_cost:
                cash_balance -= trade_cost
                new_total_investment = position['total_investment'] + (add_quantity * add_price)
                new_quantity = position['quantity'] + add_quantity
                new_avg_price = new_total_investment / new_quantity
                db.execute('UPDATE portfolios SET quantity=?, avg_price=?, total_investment=?, tranche_level=? WHERE id=?',
                           (new_quantity, new_avg_price, new_total_investment, next_tranche_level, position['id']))
                db.execute('INSERT INTO transactions (user_id, date, ticker, action, quantity, price, value) VALUES (?, ?, ?, ?, ?, ?, ?)',
                           (user_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ticker, 'ADD', add_quantity, add_price, add_quantity * add_price))
                logging.info(f"ADD: {user['username']} - {ticker} @ {add_price}")
                if user['telegram_chat_id']:
                    message = f"🔵 ADD Order Executed\n\n*Ticker:* {ticker}\n*Quantity:* {add_quantity}\n*Price:* ₹{add_price:.2f}"
                    send_telegram_message(user['telegram_chat_id'], message)

        db.execute('UPDATE users SET cash_balance = ? WHERE id = ?', (cash_balance, user_id))
        db.commit()
    except Exception as e:
        username = user['username'] if user else f"user_id:{user_id}"
        logging.error(f"CRITICAL ERROR during SELL/ADD for {username}: {e}", exc_info=True)
        db.rollback()
    finally:
        if db: db.close()

def generate_daily_watchlist(user_id):
    db = database.get_db()
    user = None
    try:
        user = db.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        if not user: return
        logging.info(f"--- Generating WATCHLIST for {user['username']} ---")
        db.execute('UPDATE users SET next_day_buy_list = NULL, daily_watchlist = NULL WHERE id = ?', (user_id,))
        db.commit()
        stock_list = [s.strip() for s in (user['stock_list'] or '').split(',') if s.strip()]
        if not stock_list: return
        portfolio_rows = db.execute('SELECT ticker FROM portfolios WHERE user_id = ?', (user_id,)).fetchall()
        portfolio = {row['ticker']: {} for row in portfolio_rows}
        
        strategy = MomentumStrategy()
        all_data = get_data_with_indicators(stock_list, '^NSEI', user['execution_interval'])
        if all_data is None: raise Exception("Failed to fetch data for watchlist generation.")
        
        buy_signals = strategy.get_buy_signals(stock_list, portfolio, all_data, user['max_open_positions'])
        watchlist_tickers = [signal['ticker'] for signal in buy_signals]
        watchlist_str = ",".join(watchlist_tickers)
        db.execute('UPDATE users SET daily_watchlist = ? WHERE id = ?', (watchlist_str, user_id))
        db.commit()
        if user['telegram_chat_id'] and watchlist_tickers:
            available_slots = max(0, user['max_open_positions'] - len(portfolio))
            tradingview_list = watchlist_str.replace('.NS', '')
            message = (f"📈 *Watchlist for Next Trading Day*\n\n"
                       f"You have *{available_slots}* position(s) available to open.\n\n"
                       f"Potential Buys:\n`{tradingview_list}`\n\n"
                       f"Please log in to the dashboard to finalize your buy list for tomorrow.")
            send_telegram_message(user['telegram_chat_id'], message)
    except Exception as e:
        username = user['username'] if user else f"user_id:{user_id}"
        logging.error(f"CRITICAL ERROR during WATCHLIST generation for {username}: {e}", exc_info=True)
        db.rollback()
    finally:
        if db: db.close()

def execute_buy_trades(user_id):
    db = database.get_db()
    user = None
    try:
        user = db.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        if not user or not user['next_day_buy_list']: return

        buy_list = [s.strip() for s in user['next_day_buy_list'].split(',') if s.strip()]
        if not buy_list: return
        
        logging.info(f"--- Executing BUYS for {user['username']}: {buy_list} ---")
        
        cash_balance = user['cash_balance']
        tranches = json.loads(user['tranche_sizes'])
        base_capital = user['base_capital']
        brokerage = user['brokerage_per_trade']
        
        for ticker in buy_list:
            if not ticker: continue
            buy_price = get_live_price(ticker)
            if buy_price is None: 
                logging.warning(f"Could not get live price for {ticker}. Skipping buy.")
                continue
            target_investment = base_capital * tranches["1"]
            quantity = round(target_investment / buy_price) if buy_price > 0 else 0
            trade_cost = (quantity * buy_price) + brokerage
            if quantity > 0 and cash_balance >= trade_cost:
                cash_balance -= trade_cost
                db.execute('INSERT INTO portfolios (user_id, ticker, quantity, avg_price, total_investment, tranche_level, entry_date) VALUES (?, ?, ?, ?, ?, ?, ?)',
                           (user_id, ticker, quantity, buy_price, quantity * buy_price, 1, datetime.now().strftime('%Y-%m-%d')))
                db.execute('INSERT INTO transactions (user_id, date, ticker, action, quantity, price, value) VALUES (?, ?, ?, ?, ?, ?, ?)',
                           (user_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ticker, 'BUY', quantity, buy_price, quantity * buy_price))
                logging.info(f"BUY: {user['username']} - {ticker} Qty: {quantity} @ {buy_price}")
                if user['telegram_chat_id']:
                    message = f"✅ BUY Order Executed\n\n*Ticker:* {ticker}\n*Quantity:* {quantity}\n*Price:* ₹{buy_price:.2f}"
                    send_telegram_message(user['telegram_chat_id'], message)
        
        db.execute('UPDATE users SET cash_balance = ? WHERE id = ?', (cash_balance, user_id))
        db.commit()
    except Exception as e:
        username = user['username'] if user else f"user_id:{user_id}"
        logging.error(f"CRITICAL ERROR during BUY execution for {username}: {e}", exc_info=True)
        db.rollback()
    finally:
        if db: db.close()

# --- BACKGROUND SCHEDULER MASTER FUNCTIONS ---
def master_scheduler_sell_add():
    with app.app_context():
        db = database.get_db()
        users = db.execute('SELECT id, username, execution_interval FROM users WHERE auto_run_enabled = 1').fetchall()
        db.close()
        today_weekday = datetime.now(pytz.timezone('Asia/Kolkata')).weekday()
        for user in users:
            if user['execution_interval'] == '1d' or (user['execution_interval'] == '1wk' and today_weekday == 4):
                execute_sell_and_add_trades(user['id'])

def master_scheduler_watchlist():
    with app.app_context():
        db = database.get_db()
        users = db.execute('SELECT id, username, execution_interval FROM users WHERE auto_run_enabled = 1').fetchall()
        db.close()
        today_weekday = datetime.now(pytz.timezone('Asia/Kolkata')).weekday()
        for user in users:
            if user['execution_interval'] == '1d' or (user['execution_interval'] == '1wk' and today_weekday == 4):
                generate_daily_watchlist(user['id'])

def master_scheduler_buy():
    with app.app_context():
        db = database.get_db()
        users = db.execute('SELECT id, username, execution_interval FROM users WHERE auto_run_enabled = 1').fetchall()
        db.close()
        today_weekday = datetime.now(pytz.timezone('Asia/Kolkata')).weekday()
        for user in users:
            if user['execution_interval'] == '1d' or (user['execution_interval'] == '1wk' and today_weekday == 0):
                 execute_buy_trades(user['id'])

@app.route('/api/save_settings', methods=['POST'])
@login_required
def save_settings():
    data = request.json
    user_id = current_user.id
    new_settings = {}
    if 'stock_list' in data:
        stock_list_str = data.get('stock_list', '')
        raw_tickers = [ticker.strip().upper() for ticker in stock_list_str.split(',') if ticker.strip()]
        if len(raw_tickers) > 200:
            return jsonify({'status': 'error', 'message': 'Stock list cannot exceed 200 tickers.'}), 400
        
        # FIX: Ensure every ticker has the .NS suffix
        processed_tickers = []
        for ticker in raw_tickers:
            if not ticker.endswith('.NS'):
                processed_tickers.append(ticker + '.NS')
            else:
                processed_tickers.append(ticker)
        
        new_settings['stock_list'] = ','.join(processed_tickers)

    if 'interval' in data:
        new_settings['execution_interval'] = data['interval']
    try:
        if 'base_capital' in data and data['base_capital']:
            new_settings['base_capital'] = float(data['base_capital'])
        if 'brokerage' in data and data['brokerage']:
            new_settings['brokerage_per_trade'] = float(data['brokerage'])
        if 'max_positions' in data and data['max_positions']:
            new_settings['max_open_positions'] = int(data['max_positions'])
    except (ValueError, TypeError):
        return jsonify({'status': 'error', 'message': 'Invalid number format for capital, brokerage, or max positions.'}), 400
    if 'tranches' in data and data['tranches']:
        try:
            tranches_obj = json.loads(data['tranches'])
            new_settings['tranche_sizes'] = json.dumps(tranches_obj)
        except json.JSONDecodeError:
            return jsonify({'status': 'error', 'message': 'Invalid JSON format for Tranches.'}), 400
    if not new_settings:
        return jsonify({'status': 'success', 'message': 'No settings were changed.'})
    set_clause = ', '.join([f"{key} = ?" for key in new_settings.keys()])
    values = list(new_settings.values())
    values.append(user_id)
    sql_query = f"UPDATE users SET {set_clause} WHERE id = ?"
    db = database.get_db()
    try:
        db.execute(sql_query, tuple(values))
        db.commit()
    except Exception as e:
        db.rollback()
        return jsonify({'status': 'error', 'message': f'Database error: {e}'}), 500
    finally:
        db.close()
    return jsonify({'status': 'success', 'message': 'Settings saved successfully!'})

@app.route('/api/save_schedule_settings', methods=['POST'])
@login_required
def save_schedule_settings():
    data = request.json
    user_id = current_user.id
    is_enabled = data.get('auto_run_enabled', False)
    db = database.get_db()
    try:
        db.execute('UPDATE users SET auto_run_enabled = ? WHERE id = ?', (is_enabled, user_id))
        db.commit()
    except Exception as e:
        db.rollback()
        return jsonify({'status': 'error', 'message': f'Database error: {e}'}), 500
    finally:
        db.close()
    return jsonify({'status': 'success', 'message': 'Schedule settings saved!'})

@app.route('/api/save_telegram_id', methods=['POST'])
@login_required
def save_telegram_id():
    chat_id = request.json.get('telegram_chat_id', '').strip()
    db = database.get_db()
    db.execute('UPDATE users SET telegram_chat_id = ? WHERE id = ?', (chat_id, current_user.id))
    db.commit()
    db.close()
    return jsonify({'status': 'success', 'message': 'Telegram Chat ID saved!'})


@app.route('/api/update_my_balance', methods=['POST'])
@login_required
def update_my_balance():
    try:
        amount_to_add = float(request.json.get('amount'))
    except (ValueError, TypeError):
        return jsonify({'status': 'error', 'message': 'Invalid amount.'}), 400
    db = database.get_db()
    db.execute('UPDATE users SET cash_balance = cash_balance + ? WHERE id = ?', (amount_to_add, current_user.id))
    db.commit()
    db.close()
    return jsonify({'status': 'success', 'message': f'₹{amount_to_add:,.2f} added to your balance.'})

@app.route('/api/change_my_password', methods=['POST'])
@login_required
def change_my_password():
    current_password = request.json.get('current_password')
    new_password = request.json.get('new_password')
    if not current_password or not new_password:
        return jsonify({'status': 'error', 'message': 'All fields are required.'}), 400
    db = database.get_db()
    user = db.execute('SELECT password_hash FROM users WHERE id = ?', (current_user.id,)).fetchone()
    if not check_password_hash(user['password_hash'], current_password):
        db.close()
        return jsonify({'status': 'error', 'message': 'Current password is not correct.'}), 400
    new_hash = generate_password_hash(new_password)
    db.execute('UPDATE users SET password_hash = ? WHERE id = ?', (new_hash, current_user.id))
    db.commit()
    db.close()
    return jsonify({'status': 'success', 'message': 'Your password has been updated.'})

@app.route('/api/run_sells_and_watchlist', methods=['POST'])
@login_required
def run_sells_and_watchlist():
    user_id = current_user.id
    logging.info(f"--- MANUAL RUN (SELLS & WATCHLIST) for user_id: {user_id} ---")
    try:
        execute_sell_and_add_trades(user_id)
        generate_daily_watchlist(user_id)
        flash("Sell/Add executed and new watchlist generated successfully.", "success")
        return jsonify({'status': 'success', 'message': 'Sell/Add executed and new watchlist generated.'})
    except Exception as e:
        logging.error(f"Error during manual sells/watchlist run for {user_id}: {e}", exc_info=True)
        flash("An error occurred during the process. Check logs.", "danger")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/execute_manual_buys', methods=['POST'])
@login_required
def execute_manual_buys():
    user_id = current_user.id
    logging.info(f"--- MANUAL RUN (BUYS) for user_id: {user_id} ---")
    try:
        execute_buy_trades(user_id)
        flash("Buy orders from your list have been executed.", "success")
        return jsonify({'status': 'success', 'message': 'Buy orders executed.'})
    except Exception as e:
        logging.error(f"Error during manual buy run for {user_id}: {e}", exc_info=True)
        flash("An error occurred during buy execution. Check logs.", "danger")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/reset', methods=['POST'])
@login_required
def reset_portfolio():
    user_id = current_user.id
    db = database.get_db()
    try:
        user = db.execute('SELECT base_capital FROM users WHERE id = ?', (user_id,)).fetchone()
        if not user:
            return jsonify({'status': 'error', 'message': 'User not found'}), 404
        new_cash_balance = user['base_capital'] * 0.40
        db.execute('UPDATE users SET cash_balance = ?, total_brokerage = 0 WHERE id = ?', (new_cash_balance, user_id))
        db.execute('DELETE FROM portfolios WHERE user_id = ?', (user_id,))
        db.execute('DELETE FROM transactions WHERE user_id = ?', (user_id,))
        db.commit()
        flash("Portfolio reset. Cash balance set to 40% of base capital.", "success")
    except Exception as e:
        db.rollback()
        flash(f"Error resetting portfolio: {e}", "danger")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        db.close()
    return jsonify({'status': 'Portfolio reset successfully.'})

# --- Stock Management Routes (Admin) ---
@app.route('/manage_stocks')
@login_required
@admin_required
def manage_stocks_page():
    db = database.get_db()
    stocks = db.execute('SELECT * FROM master_stocks ORDER BY symbol ASC').fetchall()
    db.close()
    return render_template('manage_stocks.html', stocks=stocks)

@app.route('/api/add_stock', methods=['POST'])
@login_required
@admin_required
def add_stock():
    symbol = request.form.get('symbol', '').strip().upper()
    if not symbol:
        flash("Symbol cannot be empty.", "danger")
        return redirect(url_for('manage_stocks_page'))
    if not symbol.endswith('.NS'):
        symbol += '.NS'
    db = database.get_db()
    try:
        exists = db.execute('SELECT id FROM master_stocks WHERE symbol = ?', (symbol,)).fetchone()
        if exists:
            flash(f"Stock '{symbol}' already exists.", "warning")
            return redirect(url_for('manage_stocks_page'))
        stock_info = yf.Ticker(symbol).info
        if not stock_info.get('longName'):
             flash(f"Stock '{symbol}' not found on yfinance.", "danger")
             return redirect(url_for('manage_stocks_page'))
        name = stock_info.get('longName', 'N/A')
        industry = stock_info.get('industry', 'N/A')
        sector = stock_info.get('sector', 'N/A')
        db.execute(
            'INSERT INTO master_stocks (symbol, name, industry, sector) VALUES (?, ?, ?, ?)',
            (symbol, name, industry, sector)
        )
        db.commit()
        flash(f"Stock '{symbol}' added successfully.", "success")
    except Exception as e:
        db.rollback()
        flash(f"Error adding stock '{symbol}': {e}", "danger")
    finally:
        db.close()
    return redirect(url_for('manage_stocks_page'))

@app.route('/api/add_stocks_bulk', methods=['POST'])
@login_required
@admin_required
def add_stocks_bulk():
    symbols_raw = ""
    if 'stock_file' in request.files and request.files['stock_file'].filename != '':
        file = request.files['stock_file']
        if file and file.filename.endswith('.txt'):
            symbols_raw = file.read().decode('utf-8')
        else:
            flash("Invalid file type. Please upload a .txt file.", "danger")
            return redirect(url_for('manage_stocks_page'))
    else:
        symbols_raw = request.form.get('symbol_list', '')
    if not symbols_raw:
        flash("No symbols provided.", "warning")
        return redirect(url_for('manage_stocks_page'))
    symbols = {s.strip().upper() for s in symbols_raw.replace(',', '\n').splitlines() if s.strip()}
    added_count = 0
    duplicate_count = 0
    not_found_symbols = []
    db = database.get_db()
    try:
        existing_symbols = {row['symbol'] for row in db.execute('SELECT symbol FROM master_stocks').fetchall()}
        for symbol in symbols:
            if not symbol.endswith('.NS'):
                symbol += '.NS'
            if symbol in existing_symbols:
                duplicate_count += 1
                continue
            try:
                stock_info = yf.Ticker(symbol).info
                if not stock_info.get('longName'):
                    not_found_symbols.append(symbol)
                    continue
                name = stock_info.get('longName', 'N/A')
                industry = stock_info.get('industry', 'N/A')
                sector = stock_info.get('sector', 'N/A')
                db.execute(
                    'INSERT INTO master_stocks (symbol, name, industry, sector) VALUES (?, ?, ?, ?)',
                    (symbol, name, industry, sector)
                )
                added_count += 1
            except Exception:
                not_found_symbols.append(symbol)
        db.commit()
        if added_count > 0:
            flash(f"Successfully added {added_count} new stocks.", "success")
        if duplicate_count > 0:
            flash(f"Skipped {duplicate_count} stocks that already exist in the list.", "info")
        if not_found_symbols:
            flash(f"Could not find the following symbols on yfinance: {', '.join(not_found_symbols)}", "danger")
    except Exception as e:
        db.rollback()
        flash(f"An error occurred during the bulk add process: {e}", "danger")
    finally:
        db.close()
    return redirect(url_for('manage_stocks_page'))


@app.route('/api/delete_stock', methods=['POST'])
@login_required
@admin_required
def delete_stock():
    stock_id = request.form.get('stock_id')
    db = database.get_db()
    try:
        db.execute('DELETE FROM master_stocks WHERE id = ?', (stock_id,))
        db.commit()
        flash("Stock deleted successfully.", "success")
    except Exception as e:
        db.rollback()
        flash(f"Error deleting stock: {e}", "danger")
    finally:
        db.close()
    return redirect(url_for('manage_stocks_page'))

@app.route('/api/download_stock_list')
@login_required
@admin_required
def download_stock_list():
    db = database.get_db()
    stocks = db.execute('SELECT symbol, name, industry, sector FROM master_stocks ORDER BY symbol ASC').fetchall()
    db.close()
    df = pd.DataFrame([dict(row) for row in stocks])
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='openpyxl')
    df.to_excel(writer, index=False, sheet_name='Master Stock List')
    writer.close()
    output.seek(0)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='master_stock_list.xlsx'
    )

# --- Screener Routes ---
@app.route('/screener')
@login_required
def screener_page():
    frequency = request.args.get('frequency', 'weekly')
    if frequency not in ['daily', 'weekly', 'monthly']:
        frequency = 'weekly'
    db = database.get_db()
    results = db.execute('SELECT * FROM screener_results WHERE frequency = ? ORDER BY rank ASC', (frequency,)).fetchall()
    last_run = db.execute('SELECT MAX(run_date) as last_run FROM screener_results WHERE frequency = ?', (frequency,)).fetchone()
    db.close()
    last_run_date = last_run['last_run'] if last_run and last_run['last_run'] else None
    return render_template('screener.html', results=results, last_run_date=last_run_date, current_frequency=frequency)

def run_screener_in_background(frequency='weekly'):
    with app.app_context():
        db = database.get_db()
        status_key = f'screener_status_{frequency}'
        try:
            db.execute("UPDATE app_state SET value = 'running' WHERE key = ?", (status_key,))
            db.commit()
            db.close()
            logging.info(f"Background screener process started for {frequency}.")
            stock_screener.run_screener_process(frequency=frequency)
            logging.info(f"Background screener process finished for {frequency}.")
        except Exception as e:
            logging.error(f"Error in background screener process for {frequency}: {e}", exc_info=True)
        finally:
            db_final = database.get_db()
            db_final.execute("UPDATE app_state SET value = 'idle' WHERE key = ?", (status_key,))
            db_final.commit()
            db_final.close()
            logging.info(f"Screener status for {frequency} reset to 'idle'.")

@app.route('/api/run_screener', methods=['POST'])
@login_required
@admin_required
def run_screener_api():
    frequency = request.json.get('frequency', 'weekly')
    if frequency not in ['daily', 'weekly', 'monthly']:
        return jsonify({'status': 'error', 'message': 'Invalid frequency.'}), 400
    status_key = f'screener_status_{frequency}'
    db = database.get_db()
    status_row = db.execute("SELECT value FROM app_state WHERE key = ?", (status_key,)).fetchone()
    db.close()
    if status_row and status_row['value'] == 'running':
        return jsonify({
            'status': 'error',
            'message': f'A {frequency} watchlist process is already running. Please wait for it to complete.'
        }), 409
    thread = threading.Thread(target=run_screener_in_background, args=(frequency,))
    thread.daemon = True
    thread.start()
    return jsonify({
        'status': 'success', 
        'message': f'{frequency.capitalize()} watchlist process started in the background. You will be notified when it completes.'
    })

@app.route('/api/screener_status')
@login_required
def screener_status():
    frequency = request.args.get('frequency', 'weekly')
    if frequency not in ['daily', 'weekly', 'monthly']:
        return jsonify({'status': 'unknown'})
    status_key = f'screener_status_{frequency}'
    db = database.get_db()
    status_row = db.execute("SELECT value FROM app_state WHERE key = ?", (status_key,)).fetchone()
    db.close()
    status = status_row['value'] if status_row else 'unknown'
    return jsonify({'status': status, 'frequency': frequency})

@app.route('/api/download_screener_results')
@login_required
def download_screener_results():
    frequency = request.args.get('frequency', 'weekly')
    if frequency not in ['daily', 'weekly', 'monthly']:
        frequency = 'weekly'
    db = database.get_db()
    results = db.execute('SELECT * FROM screener_results WHERE frequency = ? ORDER BY rank ASC', (frequency,)).fetchall()
    db.close()
    df = pd.DataFrame([dict(row) for row in results])
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='openpyxl')
    df.to_excel(writer, index=False, sheet_name=f'{frequency.capitalize()} Screener Results')
    writer.close()
    output.seek(0)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'stock_screener_results_{frequency}.xlsx'
    )

def scheduled_screener_job(frequency='weekly'):
    with app.app_context():
        logging.info(f"Scheduler: Running {frequency} stock screener job.")
        run_screener_in_background(frequency=frequency)
        
def update_live_prices():
    logging.info("Scheduler: Running job to update live prices.")
    with app.app_context():
        db = database.get_db()
        tickers_rows = db.execute('SELECT DISTINCT ticker FROM portfolios').fetchall()
        db.close()
        if not tickers_rows:
            logging.info("Scheduler: No tickers in any portfolio. Skipping.")
            return
        tickers = [row['ticker'] for row in tickers_rows]
        get_live_prices_bulk(tickers)


if __name__ == '__main__':
    with app.app_context():
        db = database.get_db()
        db.execute("UPDATE app_state SET value = 'idle' WHERE value = 'running'")
        db.commit()
        db.close()
        logging.info("Reset any stuck screener statuses to 'idle' on startup.")
    app.run(debug=True, port=5000, use_reloader=False)
