# app.py

import os
import json
from datetime import datetime
import pandas as pd
import yfinance as yf
import pytz
import requests
from dotenv import load_dotenv
from flask import (
    Flask, jsonify, request, render_template, redirect, url_for, flash
)
from werkzeug.security import check_password_hash, generate_password_hash
from flask_login import (
    LoginManager, UserMixin, login_user, login_required, logout_user, current_user
)
from functools import wraps
from apscheduler.schedulers.background import BackgroundScheduler

# Local imports
import database
from trading_bot import TradingBot

# --- Load Environment Variables ---
load_dotenv()

# --- App Configuration ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a-very-secret-key-that-should-be-changed'
app.config['DATABASE'] = os.path.join(app.instance_path, database.DB_NAME)
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# --- Global Cache for Live Prices ---
live_prices_cache = {}

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
        print("Telegram token or chat ID is missing. Skipping notification.")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': chat_id, 'text': message, 'parse_mode': 'Markdown'}
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            print(f"Error sending Telegram message: {response.text}")
    except Exception as e:
        print(f"Exception while sending Telegram message: {e}")

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
    
@app.route('/tcpl')
def tcpl():
    return render_template('tcplanalysis.html')
    
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

@app.route('/api/status')
@login_required
def get_status():
    db = database.get_db()
    user_id = current_user.id
    user_data = db.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    portfolio_rows = db.execute('SELECT * FROM portfolios WHERE user_id = ?', (user_id,)).fetchall()
    transaction_rows = db.execute('SELECT * FROM transactions WHERE user_id = ? ORDER BY date DESC', (user_id,)).fetchall()
    db.close()

    portfolio = {row['ticker']: {key: row[key] for key in row.keys()} for row in portfolio_rows}
    transactions = [{key: row[key] for key in row.keys()} for row in transaction_rows]
    
    # --- On-demand price fetching for tickers not in cache ---
    tickers_to_fetch_now = [ticker for ticker in portfolio.keys() if ticker not in live_prices_cache]
    if tickers_to_fetch_now:
        print(f"get_status: Fetching live prices for {tickers_to_fetch_now}")
        try:
            live_data = yf.download(tickers_to_fetch_now, period='1d', progress=False)
            if not live_data.empty:
                close_prices = live_data['Close']
                if isinstance(close_prices, pd.Series): # Handle single ticker case
                    last_prices = {tickers_to_fetch_now[0]: close_prices.iloc[-1]}
                else: # Multiple tickers
                    last_prices = close_prices.iloc[-1].to_dict()
                
                for ticker, price in last_prices.items():
                    if pd.notna(price):
                        live_prices_cache[ticker] = price
        except Exception as e:
            print(f"get_status: Error fetching prices on-demand: {e}")

    unrealized_pnl = 0
    holdings_value = 0
    for ticker, pos in portfolio.items():
        last_price = live_prices_cache.get(ticker)
        if last_price:
            pos['current_price'] = last_price
            pos['market_value'] = pos['quantity'] * last_price
            pos['pnl'] = (last_price - pos['avg_price']) * pos['quantity']
            unrealized_pnl += pos['pnl']
            holdings_value += pos['market_value']
        else:
            pos['current_price'] = 0
            pos['market_value'] = 0
            pos['pnl'] = 0

    portfolio_value = holdings_value + user_data['cash_balance']

    realized_pnl = 0
    closed_trades = {}
    daily_pnl = {}
    
    for tx in reversed(transactions):
        ticker = tx['ticker']
        if tx['action'] == 'BUY' or tx['action'] == 'ADD':
            if ticker not in closed_trades:
                closed_trades[ticker] = {'quantity': 0, 'cost': 0}
            closed_trades[ticker]['quantity'] += tx['quantity']
            closed_trades[ticker]['cost'] += tx['value']
        
        elif tx['action'] == 'SELL':
            if ticker in closed_trades and closed_trades[ticker]['quantity'] > 0:
                sell_qty = tx['quantity']
                avg_buy_price = closed_trades[ticker]['cost'] / closed_trades[ticker]['quantity']
                cost_of_sold_shares = sell_qty * avg_buy_price
                pnl = tx['value'] - cost_of_sold_shares
                realized_pnl += pnl
                trade_date = datetime.strptime(tx['date'], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
                daily_pnl[trade_date] = daily_pnl.get(trade_date, 0) + pnl
                closed_trades[ticker]['quantity'] -= sell_qty
                closed_trades[ticker]['cost'] -= cost_of_sold_shares

    wins = sum(1 for pnl in daily_pnl.values() if pnl > 0)
    losses = sum(1 for pnl in daily_pnl.values() if pnl < 0)
    total_closed = wins + losses
    win_ratio = (wins / total_closed * 100) if total_closed > 0 else 0
    avg_win = sum(pnl for pnl in daily_pnl.values() if pnl > 0) / wins if wins > 0 else 0
    avg_loss = sum(pnl for pnl in daily_pnl.values() if pnl < 0) / losses if losses > 0 else 0

    return jsonify({
        'portfolio': portfolio,
        'transactions': transactions,
        'settings': {key: user_data[key] for key in user_data.keys()},
        'stats': {
            'portfolio_value': portfolio_value,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': realized_pnl,
            'win_ratio': win_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
        },
        'calendar_pnl': daily_pnl
    })

@app.route('/api/save_settings', methods=['POST'])
@login_required
def save_settings():
    data = request.json
    user_id = current_user.id
    
    new_settings = {}

    if 'stock_list' in data:
        stock_list_str = data.get('stock_list', '')
        stock_list = [ticker.strip().upper() for ticker in stock_list_str.split(',') if ticker.strip()]
        if len(stock_list) > 200:
            return jsonify({'status': 'error', 'message': 'Stock list cannot exceed 200 tickers.'}), 400
        new_settings['stock_list'] = ','.join(stock_list)

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


@app.route('/api/run_strategy', methods=['POST'])
@login_required
def run_strategy():
    user_id = current_user.id
    log = []
    
    def _log_message(message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        full_message = f"[{timestamp}] {message}"
        print(full_message)
        log.append(full_message)

    if not is_market_open() and current_user.role != 'admin':
        _log_message("Market is closed.")
        return jsonify({'status': 'error', 'log': log})

    db = database.get_db()
    try:
        user = db.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        portfolio_rows = db.execute('SELECT * FROM portfolios WHERE user_id = ?', (user_id,)).fetchall()
        
        stock_list = [s.strip() for s in (user['stock_list'] or '').split(',') if s.strip()]
        if not stock_list:
            _log_message("Error: Stock list is not configured.")
            return jsonify({'status': 'error', 'log': log})

        max_open_positions = user['max_open_positions']
        brokerage = user['brokerage_per_trade']
        base_capital = user['base_capital']
        tranches = json.loads(user['tranche_sizes'])
        cash_balance = user['cash_balance']
        
        portfolio = {row['ticker']: dict(row) for row in portfolio_rows}

        bot = TradingBot(stock_tickers=stock_list, benchmark_ticker='^NSEI', interval=user['execution_interval'])
        all_data, fetch_log = bot.get_analysis()
        log.extend(fetch_log)

        if all_data is None:
            raise Exception("Failed to fetch market data.")

        _log_message("\n--- Checking Sell Conditions ---")
        tickers_to_sell = []
        total_brokerage_session = 0
        for ticker, position in portfolio.items():
            stock_data = all_data.get(ticker)
            if stock_data is None or stock_data.empty:
                _log_message(f"No data for {ticker}, cannot evaluate.")
                continue

            latest_data = stock_data.iloc[-1]
            sell_signal = False
            reason = ""

            if position['tranche_level'] == 1:
                stop_loss_price = position['avg_price'] * 0.90
                if latest_data['Close'] < stop_loss_price:
                    sell_signal = True
                    reason = f"10% stop-loss (Price < ₹{stop_loss_price:.2f})"

            if not sell_signal and position['tranche_level'] >= 3:
                if 'EMA_11' in latest_data and latest_data['Close'] < latest_data['EMA_11']:
                    sell_signal = True
                    reason = "Price crossed below EMA_11"
            
            if not sell_signal:
                if 'EMA_21' in latest_data and latest_data['Close'] < latest_data['EMA_21']:
                    sell_signal = True
                    reason = "Price crossed below EMA_21"

            if sell_signal:
                tickers_to_sell.append({'ticker': ticker, 'reason': reason})
        
        if tickers_to_sell:
            for sell_info in tickers_to_sell:
                ticker = sell_info['ticker']
                reason = sell_info['reason']
                position = portfolio[ticker]
                sell_price = all_data.get(ticker).iloc[-1]['Close']
                sell_value = position['quantity'] * sell_price
                cash_balance += sell_value - brokerage
                total_brokerage_session += brokerage
                
                _log_message(f"SELL: {ticker} Qty: {position['quantity']} @ ₹{sell_price:.2f}. Reason: {reason}")
                db.execute('DELETE FROM portfolios WHERE user_id = ? AND ticker = ?', (user_id, ticker))
                db.execute('INSERT INTO transactions (user_id, date, ticker, action, quantity, price, value) VALUES (?, ?, ?, ?, ?, ?, ?)',
                           (user_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ticker, 'SELL', position['quantity'], sell_price, sell_value))
                
                if user['telegram_chat_id']:
                    message = f"🔴 SELL Order Executed\n\n*Ticker:* {ticker}\n*Quantity:* {position['quantity']}\n*Price:* ₹{sell_price:.2f}\n*Reason:* {reason}"
                    send_telegram_message(user['telegram_chat_id'], message)

                del portfolio[ticker]

        _log_message("\n--- Checking Add-on Conditions ---")
        for ticker, position in portfolio.items():
            if str(position['tranche_level'] + 1) not in tranches:
                continue

            stock_data = all_data.get(ticker)
            if stock_data is None or stock_data.empty: continue
            
            latest_data = stock_data.iloc[-1]
            wick_ratio = (latest_data['High'] - latest_data['Close']) / (latest_data['High'] - latest_data['Low'] + 1e-6)

            if latest_data['High'] >= latest_data['DCU_20_20'] and wick_ratio < 0.3:
                next_tranche_level = position['tranche_level'] + 1
                target_investment = base_capital * tranches[str(next_tranche_level)]
                add_price = latest_data['Close']
                add_quantity = round(target_investment / add_price) if add_price > 0 else 0
                trade_cost = (add_quantity * add_price) + brokerage

                if add_quantity > 0 and cash_balance >= trade_cost:
                    cash_balance -= trade_cost
                    total_brokerage_session += brokerage
                    _log_message(f"ADD: {ticker} Qty: {add_quantity} @ ₹{add_price:.2f}")
                    
                    new_total_investment = position['total_investment'] + (add_quantity * add_price)
                    new_quantity = position['quantity'] + add_quantity
                    new_avg_price = new_total_investment / new_quantity
                    
                    db.execute('UPDATE portfolios SET quantity=?, avg_price=?, total_investment=?, tranche_level=? WHERE id=?',
                               (new_quantity, new_avg_price, new_total_investment, next_tranche_level, position['id']))
                    db.execute('INSERT INTO transactions (user_id, date, ticker, action, quantity, price, value) VALUES (?, ?, ?, ?, ?, ?, ?)',
                               (user_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ticker, 'ADD', add_quantity, add_price, add_quantity * add_price))
                    
                    if user['telegram_chat_id']:
                        message = f"🔵 ADD Order Executed\n\n*Ticker:* {ticker}\n*Quantity:* {add_quantity}\n*Price:* ₹{add_price:.2f}"
                        send_telegram_message(user['telegram_chat_id'], message)

                    portfolio[ticker]['quantity'] = new_quantity
                    
        _log_message("\n--- Checking Buy Conditions ---")
        if len(portfolio) < max_open_positions:
            for ticker in stock_list:
                if len(portfolio) >= max_open_positions:
                    _log_message("Max open positions reached.")
                    break
                if ticker in portfolio: continue

                stock_data = all_data.get(ticker)
                if stock_data is None or stock_data.empty: continue
                
                latest_data = stock_data.iloc[-1]
                
                is_ema_crossover = latest_data['EMA_11'] > latest_data['EMA_21']
                is_strong_momentum = latest_data['RS'] > 1.0 and latest_data['ADX_14'] > 25 and latest_data['RSI_14'] > 55
                is_volume_spike = latest_data['Volume'] > (1.25 * latest_data.get('Volume_MA10', 0))
                
                if is_ema_crossover and is_strong_momentum and is_volume_spike:
                    target_investment = base_capital * tranches["1"]
                    buy_price = latest_data['Close']
                    quantity = round(target_investment / buy_price) if buy_price > 0 else 0
                    trade_cost = (quantity * buy_price) + brokerage
                    
                    if quantity > 0 and cash_balance >= trade_cost:
                        cash_balance -= trade_cost
                        total_brokerage_session += brokerage
                        _log_message(f"BUY: {ticker} Qty: {quantity} @ ₹{buy_price:.2f}")
                        
                        db.execute('INSERT INTO portfolios (user_id, ticker, quantity, avg_price, total_investment, tranche_level, entry_date) VALUES (?, ?, ?, ?, ?, ?, ?)',
                                   (user_id, ticker, quantity, buy_price, quantity * buy_price, 1, datetime.now().strftime('%Y-%m-%d')))
                        db.execute('INSERT INTO transactions (user_id, date, ticker, action, quantity, price, value) VALUES (?, ?, ?, ?, ?, ?, ?)',
                                   (user_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ticker, 'BUY', quantity, buy_price, quantity * buy_price))
                        
                        if user['telegram_chat_id']:
                            message = f"✅ BUY Order Executed\n\n*Ticker:* {ticker}\n*Quantity:* {quantity}\n*Price:* ₹{buy_price:.2f}"
                            send_telegram_message(user['telegram_chat_id'], message)

                        portfolio[ticker] = {'ticker': ticker}

        _log_message("\n--- Strategy Execution Finished ---")
        db.execute('UPDATE users SET cash_balance = ?, total_brokerage = total_brokerage + ? WHERE id = ?', (cash_balance, total_brokerage_session, user_id))
        db.commit()
    except Exception as e:
        db.rollback()
        _log_message(f"CRITICAL ERROR: {e}")
    finally:
        db.close()
    
    return jsonify({'status': 'Strategy execution finished.', 'log': log})


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


# --- Background Scheduler for P&L Updates ---
def update_live_prices():
    """Scheduled job to fetch live prices for all tickers in portfolios."""
    print("Scheduler: Running job to update live prices.")
    with app.app_context():
        db = database.get_db()
        tickers_rows = db.execute('SELECT DISTINCT ticker FROM portfolios').fetchall()
        db.close()
        
        if not tickers_rows:
            print("Scheduler: No tickers in any portfolio. Skipping.")
            return

        tickers = [row['ticker'] for row in tickers_rows]
        try:
            live_data = yf.download(tickers, period='1d', progress=False)
            if live_data.empty:
                print("Scheduler: No data from yfinance.")
                return
            
            close_prices = live_data['Close']
            if isinstance(close_prices, pd.Series):
                last_prices = {tickers[0]: close_prices.iloc[-1]}
            else:
                last_prices = close_prices.iloc[-1].to_dict()

            for ticker, price in last_prices.items():
                if pd.notna(price):
                    live_prices_cache[ticker] = price
            print(f"Scheduler: Updated prices for: {list(live_prices_cache.keys())}")

        except Exception as e:
            print(f"Scheduler: Error fetching prices: {e}")

if __name__ == '__main__':
    scheduler = BackgroundScheduler(daemon=True, timezone='Asia/Kolkata')
    scheduler.add_job(update_live_prices, 'cron', day_of_week='mon-fri', hour='9-16', minute='30')
    scheduler.start()
    
    with app.app_context():
        update_live_prices()

    app.run(debug=True, port=5000)
