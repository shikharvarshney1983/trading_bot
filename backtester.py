# backtester.py
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
from collections import deque
from data_utils import get_data_with_indicators # Use the centralized data utility

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
        self.strategy = strategy
        
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

    def run(self):
        """Runs the backtest simulation day by day."""
        self._log("Fetching backtest data and calculating indicators...")
        all_indicator_data = get_data_with_indicators(
            tickers=self.stock_tickers,
            benchmark_ticker='^NSEI',
            interval=self.interval
        )
        if all_indicator_data is None:
            raise ValueError("Failed to fetch data for backtest.")

        self._log("Starting backtest simulation...")
        
        all_dates = pd.Index([])
        for ticker in self.stock_tickers:
            if ticker in all_indicator_data:
                all_dates = all_dates.union(all_indicator_data[ticker].index)
        
        simulation_dates = all_dates[(all_dates >= self.start_date) & (all_dates <= self.end_date)].sort_values()

        for date in simulation_dates:
            current_data_snapshot = {
                ticker: df[df.index <= date] 
                for ticker, df in all_indicator_data.items() 
                if ticker in all_indicator_data and not df.empty
            }

            current_holdings_value = 0
            for ticker, pos in self.portfolio.items():
                if ticker in current_data_snapshot and not current_data_snapshot[ticker].empty:
                    price = current_data_snapshot[ticker].iloc[-1]['Close']
                    current_holdings_value += pos['quantity'] * price
                else:
                    current_holdings_value += pos['quantity'] * pos['avg_price']
            
            self.portfolio_value_history.append({
                'date': date.strftime('%Y-%m-%d'),
                'value': self.cash + current_holdings_value
            })

            sell_signals = self.strategy.get_sell_signals(self.portfolio, current_data_snapshot)
            add_on_signals = self.strategy.get_add_on_signals(self.portfolio, current_data_snapshot)
            buy_signals = self.strategy.get_buy_signals(self.stock_tickers, self.portfolio, current_data_snapshot, self.max_positions)

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
        """Generates a summary of the backtest performance with accurate FIFO P&L."""
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

        # FIX: Accurate Realized P&L Calculation using FIFO
        trades_df = pd.DataFrame(self.transactions)
        investments = {} # Using a deque to store lots
        realized_pnl = 0
        pnl_list = []

        for _, row in trades_df.sort_values(by='date').iterrows():
            ticker = row['ticker']
            if ticker not in investments:
                investments[ticker] = deque()

            if row['action'] in ['BUY', 'ADD']:
                investments[ticker].append({'quantity': row['quantity'], 'price': row['price']})
            
            elif row['action'] == 'SELL':
                sell_quantity = row['quantity']
                sell_value = row['value']
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
                
                profit = sell_value - cost_of_sold_shares
                realized_pnl += profit
                pnl_list.append(profit)

        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p < 0]
        win_ratio = (len(wins) / len(pnl_list) * 100) if pnl_list else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        unrealized_pnl = 0
        final_portfolio_summary = []
        last_date_str = self.portfolio_value_history[-1]['date']
        last_date_dt = pd.to_datetime(last_date_str)

        for ticker, pos in self.portfolio.items():
            last_price = pos['avg_price']
            if ticker in all_indicator_data:
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
