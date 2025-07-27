# backtester.py
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timedelta
import json
import logging
import numpy as np

class Backtester:
    def __init__(self, stock_tickers, start_date, end_date, interval, initial_capital, strategy_capital, tranche_sizes_pct, brokerage, max_positions, strategy):
        self.stock_tickers = stock_tickers
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.initial_capital = initial_capital
        self.strategy_capital = strategy_capital
        self.tranche_sizes_pct = tranche_sizes_pct
        self.brokerage = brokerage
        self.max_positions = max_positions
        self.strategy = strategy # The strategy object to use for signals
        
        self.cash = initial_capital
        self.portfolio = {}
        self.transactions = []
        self.portfolio_value_history = []
        self.log = []
        logging.basicConfig(filename='trading_bot.log', level=logging.INFO)

    def _log(self, message):
        """Logs a message and prints it for immediate feedback during backtesting."""
        print(message)
        self.log.append(message)
        logging.info(message)

    def _fetch_data(self):
        """Fetches all necessary historical data from yfinance."""
        self._log("Fetching backtest data...")
        all_tickers = self.stock_tickers + ['^NSEI']
        # Fetch more historical data to ensure indicators are calculated correctly from the start
        buffer_start_date = self.start_date - timedelta(days=300)
        
        data = yf.download(all_tickers, start=buffer_start_date, end=self.end_date + timedelta(days=1), interval=self.interval, progress=False, auto_adjust=True)
        
        if data.empty:
            raise ValueError("No data downloaded for the given tickers and date range.")
        
        # Standardize column access
        data.columns = data.columns.swaplevel(0, 1)
        ticker_data = {}
        for ticker in all_tickers:
            if ticker in data and not data[ticker].empty:
                df = data[ticker].copy()
                # No need to rename columns with auto_adjust=True
                ticker_data[ticker] = df

        return ticker_data

    def _calculate_indicators(self, data):
        """Calculates all indicators needed for the strategy across all stocks."""
        self._log("Calculating indicators for all stocks...")
        if '^NSEI' not in data:
            raise ValueError("Benchmark data (^NSEI) could not be fetched.")
        
        benchmark_data = data['^NSEI']

        for ticker in self.stock_tickers:
            if ticker not in data: continue
            stock_data = data[ticker]
            
            # Ensure required columns are present
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in stock_data.columns for col in required_cols):
                self._log(f"Warning: Skipping {ticker} due to missing OHLCV data.")
                continue

            # Calculate all indicators required by the MomentumStrategy
            stock_data.ta.ema(length=11, append=True)
            stock_data.ta.ema(length=21, append=True)
            stock_data.ta.ema(length=50, append=True)
            stock_data.ta.rsi(length=14, append=True)
            stock_data.ta.adx(length=14, append=True)
            stock_data.ta.donchian(lower_length=20, upper_length=20, append=True)
            stock_data['Volume_MA10'] = stock_data['Volume'].rolling(window=10).mean()

            # Calculate Relative Strength (RS)
            roll_period = 50 if self.interval == '1wk' else 10
            stock_ret = stock_data['Close'].pct_change().rolling(roll_period).sum()
            bench_ret = benchmark_data['Close'].pct_change().rolling(roll_period).sum()
            stock_data['RS'] = stock_ret / bench_ret
            
            data[ticker] = stock_data.dropna()
        
        return data

    def run(self):
        """Runs the backtest simulation day by day."""
        all_ticker_data = self._fetch_data()
        all_indicator_data = self._calculate_indicators(all_ticker_data)
        
        self._log("Starting backtest simulation...")
        
        # Create a unified date index for the simulation loop
        all_dates = pd.Index([])
        for ticker in self.stock_tickers:
            if ticker in all_indicator_data:
                all_dates = all_dates.union(all_indicator_data[ticker].index)
        
        # Filter dates to be within the specified backtest range
        simulation_dates = all_dates[(all_dates >= self.start_date) & (all_dates <= self.end_date)]

        for date in simulation_dates:
            # Create a snapshot of data up to the current simulation date for the strategy
            current_data_snapshot = {
                ticker: df[df.index <= date] 
                for ticker, df in all_indicator_data.items() 
                if ticker in all_indicator_data and not df.empty
            }

            # --- Record Portfolio Value for this period ---
            current_holdings_value = 0
            for ticker, pos in self.portfolio.items():
                if ticker in current_data_snapshot and not current_data_snapshot[ticker].empty:
                    price = current_data_snapshot[ticker].iloc[-1]['Close']
                    current_holdings_value += pos['quantity'] * price
                else: # If no current price, use average price (should be rare)
                    current_holdings_value += pos['quantity'] * pos['avg_price']
            
            self.portfolio_value_history.append({
                'date': date.strftime('%Y-%m-%d'),
                'value': self.cash + current_holdings_value
            })

            # --- Get Signals from Strategy ---
            sell_signals = self.strategy.get_sell_signals(self.portfolio, current_data_snapshot)
            add_on_signals = self.strategy.get_add_on_signals(self.portfolio, current_data_snapshot)
            buy_signals = self.strategy.get_buy_signals(self.stock_tickers, self.portfolio, current_data_snapshot, self.max_positions)

            # --- Execute Trades ---
            # 1. Sells
            for signal in sell_signals:
                ticker = signal['ticker']
                position = self.portfolio[ticker]
                sell_price = current_data_snapshot[ticker].iloc[-1]['Close']
                sell_value = position['quantity'] * sell_price
                self.cash += sell_value - self.brokerage
                
                self.transactions.append({
                    'date': date.strftime('%Y-%m-%d'), 'ticker': ticker, 'action': 'SELL',
                    'quantity': position['quantity'], 'price': sell_price, 'value': sell_value
                })
                del self.portfolio[ticker]

            # 2. Add-ons
            for signal in add_on_signals:
                ticker = signal['ticker']
                position = self.portfolio[ticker]
                next_tranche_level = position['tranche_level'] + 1
                
                if str(next_tranche_level) not in self.tranche_sizes_pct: continue

                investment_amount = self.strategy_capital * self.tranche_sizes_pct[str(next_tranche_level)]
                add_price = signal['price']
                quantity = int(investment_amount / add_price) if add_price > 0 else 0
                cost = (quantity * add_price) + self.brokerage

                if quantity > 0 and self.cash >= cost:
                    self.cash -= cost
                    
                    new_total_investment = position['total_investment'] + (quantity * add_price)
                    new_quantity = position['quantity'] + quantity
                    new_avg_price = new_total_investment / new_quantity
                    
                    position['quantity'] = new_quantity
                    position['avg_price'] = new_avg_price
                    position['total_investment'] = new_total_investment
                    position['tranche_level'] = next_tranche_level

                    self.transactions.append({
                        'date': date.strftime('%Y-%m-%d'), 'ticker': ticker, 'action': 'ADD',
                        'quantity': quantity, 'price': add_price, 'value': quantity * add_price
                    })

            # 3. Buys
            for signal in buy_signals:
                ticker = signal['ticker']
                investment_amount = self.strategy_capital * self.tranche_sizes_pct["1"]
                buy_price = signal['price']
                quantity = int(investment_amount / buy_price) if buy_price > 0 else 0
                cost = (quantity * buy_price) + self.brokerage
                
                if quantity > 0 and self.cash >= cost:
                    self.cash -= cost
                    self.portfolio[ticker] = {
                        'quantity': quantity, 
                        'avg_price': buy_price,
                        'total_investment': quantity * buy_price, 
                        'tranche_level': 1
                    }
                    self.transactions.append({
                        'date': date.strftime('%Y-%m-%d'), 'ticker': ticker, 'action': 'BUY',
                        'quantity': quantity, 'price': buy_price, 'value': quantity * buy_price
                    })
        
        self._log("Backtest simulation finished.")
        return self._generate_results(all_indicator_data)

    def _generate_results(self, all_indicator_data):
        """Generates a summary of the backtest performance."""
        if not self.portfolio_value_history:
            return {"summary": {"error": "No data to generate results."}, "log": self.log}

        final_value = self.portfolio_value_history[-1]['value']
        total_return_pct = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        num_years = (self.end_date - self.start_date).days / 365.25
        cagr = ((final_value / self.initial_capital) ** (1 / num_years)) - 1 if num_years > 0 and self.initial_capital > 0 else 0
        
        df = pd.DataFrame(self.portfolio_value_history)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        rolling_max = df['value'].cummax()
        daily_drawdown = df['value'] / rolling_max - 1.0
        max_drawdown = daily_drawdown.min() * 100

        # Calculate realized P&L from transactions
        trades_df = pd.DataFrame(self.transactions)
        pnl_list, realized_pnl = [], 0
        temp_portfolio = {}
        for _, row in trades_df.iterrows():
            ticker = row['ticker']
            if row['action'] in ['BUY', 'ADD']:
                if ticker not in temp_portfolio:
                    temp_portfolio[ticker] = {'cost': 0, 'qty': 0}
                temp_portfolio[ticker]['cost'] += row['value']
                temp_portfolio[ticker]['qty'] += row['quantity']
            elif row['action'] == 'SELL':
                if ticker in temp_portfolio and temp_portfolio[ticker]['qty'] > 0:
                    avg_cost = temp_portfolio[ticker]['cost'] / temp_portfolio[ticker]['qty'] if temp_portfolio[ticker]['qty'] > 0 else 0
                    cost_of_sold = row['quantity'] * avg_cost
                    profit = row['value'] - cost_of_sold
                    pnl_list.append(profit)
                    realized_pnl += profit
                    temp_portfolio[ticker]['cost'] -= cost_of_sold
                    temp_portfolio[ticker]['qty'] -= row['quantity']
        
        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p < 0]
        win_ratio = (len(wins) / len(pnl_list) * 100) if pnl_list else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        # Calculate unrealized P&L for final holdings
        unrealized_pnl = 0
        final_portfolio_summary = []
        last_date_str = self.portfolio_value_history[-1]['date']
        # FIX: The yfinance index is timezone-naive, so the comparison datetime should also be naive.
        # The .tz_localize() call was causing the error.
        last_date_dt = datetime.strptime(last_date_str, '%Y-%m-%d')

        for ticker, pos in self.portfolio.items():
            last_price = pos['avg_price'] # Default to avg price
            if ticker in all_indicator_data:
                # Filter data up to the last date of the simulation
                relevant_prices = all_indicator_data[ticker][all_indicator_data[ticker].index <= last_date_dt]
                if not relevant_prices.empty:
                    last_price = relevant_prices.iloc[-1]['Close']
            
            pnl_val = (last_price - pos['avg_price']) * pos['quantity']
            unrealized_pnl += pnl_val
            final_portfolio_summary.append({
                'ticker': ticker, 'quantity': pos['quantity'], 'avg_price': pos['avg_price'],
                'last_price': last_price, 'pnl': pnl_val
            })

        return {
            "summary": {
                "start_date": self.start_date.strftime('%Y-%m-%d'),
                "end_date": self.end_date.strftime('%Y-%m-%d'),
                "initial_capital": self.initial_capital,
                "final_portfolio_value": final_value,
                "total_return_pct": total_return_pct,
                "cagr_pct": cagr * 100,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "max_drawdown_pct": max_drawdown,
                "total_trades": len(pnl_list),
                "win_ratio_pct": win_ratio,
                "avg_win": avg_win,
                "avg_loss": avg_loss
            },
            "pnl_history": self.portfolio_value_history,
            "transactions": self.transactions,
            "final_portfolio": final_portfolio_summary,
            "log": self.log
        }
