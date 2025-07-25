# backtester.py
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timedelta
import json
import logging
import numpy as np

class Backtester:
    def __init__(self, stock_tickers, start_date, end_date, interval, initial_capital, tranche_sizes_pct, brokerage, max_positions):
        self.stock_tickers = stock_tickers
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.initial_capital = initial_capital
        self.tranche_sizes_pct = tranche_sizes_pct
        self.brokerage = brokerage
        self.max_positions = max_positions
        
        self.cash = initial_capital
        self.portfolio = {}
        self.transactions = []
        self.daily_pnl = {}
        self.portfolio_value_history = []
        self.log = []
        logging.basicConfig(filename='trading_bot.log', level=logging.INFO)

    def _log(self, message):
        self.log.append(message)
        logging.info(message)

    def _fetch_data(self):
        self._log("Fetching backtest data...")
        all_tickers = self.stock_tickers + ['^NSEI']
        buffer_start_date = self.start_date - timedelta(days=200)
        
        data = yf.download(all_tickers, start=buffer_start_date, end=self.end_date + timedelta(days=1), interval=self.interval, progress=False, group_by='ticker')
        
        if data.empty:
            raise ValueError("No data downloaded for the given tickers and date range.")
        
        ticker_data = {}
        for ticker in all_tickers:
            if ticker in data and not data[ticker].empty:
                df = data[ticker].copy()
                df.rename(columns=str.lower, inplace=True)
                ticker_data[ticker] = df

        return ticker_data

    def _calculate_indicators(self, data):
        self._log("Calculating indicators for all stocks...")
        if '^NSEI' not in data:
            raise ValueError("Benchmark data (^NSEI) could not be fetched.")
        
        benchmark_data = data['^NSEI']

        for ticker in self.stock_tickers:
            if ticker not in data: continue
            stock_data = data[ticker]
            
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in stock_data.columns for col in required_cols):
                self._log(f"Warning: Skipping {ticker} due to missing OHLCV data.")
                continue

            stock_data.ta.ema(length=11, append=True)
            stock_data.ta.ema(length=21, append=True)
            stock_data.ta.ema(length=40, append=True)
            stock_data.ta.rsi(length=14, append=True)
            stock_data.ta.adx(length=14, append=True)
            stock_data['volume_ma10'] = stock_data['volume'].rolling(window=10).mean()

            roll_period = 50 if self.interval == '1wk' else 10
            stock_ret = stock_data['close'].pct_change().rolling(roll_period).sum()
            bench_ret = benchmark_data['close'].pct_change().rolling(roll_period).sum()
            stock_data['rs'] = stock_ret / bench_ret
            
            data[ticker] = stock_data.dropna()
        
        return data

    def run(self):
        all_ticker_data = self._fetch_data()
        all_indicator_data = self._calculate_indicators(all_ticker_data)
        
        self._log("Starting backtest simulation...")
        
        all_dates = sorted(list(set(date for ticker in self.stock_tickers if ticker in all_indicator_data for date in all_indicator_data[ticker].index)))
        
        for date in all_dates:
            if date.to_pydatetime() < self.start_date:
                continue

            current_holdings_value = 0
            for ticker, pos in self.portfolio.items():
                if ticker in all_indicator_data and date in all_indicator_data[ticker].index:
                    price = all_indicator_data[ticker].loc[date]['close']
                    current_holdings_value += pos['quantity'] * price
                else:
                    current_holdings_value += pos['quantity'] * pos['avg_price']
            
            self.portfolio_value_history.append({
                'date': date.strftime('%Y-%m-%d'),
                'value': self.cash + current_holdings_value
            })

            tickers_to_sell = []
            for ticker, position in self.portfolio.items():
                if ticker not in all_indicator_data or date not in all_indicator_data[ticker].index: continue
                
                stock_row = all_indicator_data[ticker].loc[date]
                sell_signal = False
                # FIX: Use uppercase indicator names
                if position['tranche_level'] == 1 and stock_row['close'] < position['avg_price'] * 0.90:
                    sell_signal = True
                elif position['tranche_level'] >= 3 and stock_row['close'] < stock_row['EMA_11']:
                    sell_signal = True
                elif stock_row['close'] < stock_row['EMA_21']:
                    sell_signal = True
                
                if sell_signal:
                    tickers_to_sell.append(ticker)

            for ticker in tickers_to_sell:
                position = self.portfolio[ticker]
                sell_price = all_indicator_data[ticker].loc[date]['close']
                sell_value = position['quantity'] * sell_price
                self.cash += sell_value - self.brokerage
                
                self.transactions.append({
                    'date': date.strftime('%Y-%m-%d'), 'ticker': ticker, 'action': 'SELL',
                    'quantity': position['quantity'], 'price': sell_price, 'value': sell_value
                })
                del self.portfolio[ticker]

            if len(self.portfolio) < self.max_positions:
                for ticker in self.stock_tickers:
                    if ticker in self.portfolio or ticker not in all_indicator_data: continue
                    
                    stock_df = all_indicator_data[ticker]
                    if date not in stock_df.index: continue
                    
                    idx = stock_df.index.get_loc(date)
                    if idx < 1: continue
                    
                    latest_data = stock_df.iloc[idx]
                    previous_data = stock_df.iloc[idx - 1]
                    
                    is_making_higher_close = latest_data['close'] > previous_data['close']
                    intraday_move_pct = ((latest_data['close'] - latest_data['open']) / latest_data['open']) * 100 if latest_data['open'] > 0 else 0
                    is_not_major_reversal = intraday_move_pct > -1.0
                    
                    # FIX: Use uppercase indicator names
                    is_ema_uptrend = latest_data['close'] > latest_data['EMA_11'] > latest_data['EMA_21'] > latest_data['EMA_40']
                    is_strong_momentum = latest_data['rs'] > 1.0 and latest_data['ADX_14'] > 25 and latest_data['RSI_14'] > 55
                    is_volume_spike = latest_data['volume'] > (1.25 * latest_data.get('volume_ma10', 0))

                    if is_ema_uptrend and is_strong_momentum and is_volume_spike and is_making_higher_close and is_not_major_reversal:
                        investment_amount = self.initial_capital * self.tranche_sizes_pct["1"]
                        quantity = int(investment_amount / latest_data['close'])
                        cost = (quantity * latest_data['close']) + self.brokerage
                        
                        if quantity > 0 and self.cash >= cost:
                            self.cash -= cost
                            self.portfolio[ticker] = {
                                'quantity': quantity, 'avg_price': latest_data['close'],
                                'total_investment': quantity * latest_data['close'], 'tranche_level': 1
                            }
                            self.transactions.append({
                                'date': date.strftime('%Y-%m-%d'), 'ticker': ticker, 'action': 'BUY',
                                'quantity': quantity, 'price': latest_data['close'], 'value': quantity * latest_data['close']
                            })
                            if len(self.portfolio) >= self.max_positions:
                                break
        
        self._log("Backtest simulation finished.")
        return self._generate_results(all_indicator_data)

    def _generate_results(self, all_indicator_data):
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

        trades_df = pd.DataFrame(self.transactions)
        pnl, realized_pnl = [], 0
        temp_portfolio = {}
        for _, row in trades_df.iterrows():
            if row['action'] == 'BUY':
                if row['ticker'] not in temp_portfolio:
                    temp_portfolio[row['ticker']] = {'cost': 0, 'qty': 0}
                temp_portfolio[row['ticker']]['cost'] += row['value']
                temp_portfolio[row['ticker']]['qty'] += row['quantity']
            elif row['action'] == 'SELL':
                if row['ticker'] in temp_portfolio and temp_portfolio[row['ticker']]['qty'] > 0:
                    avg_cost = temp_portfolio[row['ticker']]['cost'] / temp_portfolio[row['ticker']]['qty'] if temp_portfolio[row['ticker']]['qty'] > 0 else 0
                    cost_of_sold = row['quantity'] * avg_cost
                    profit = row['value'] - cost_of_sold
                    pnl.append(profit)
                    realized_pnl += profit
                    temp_portfolio[row['ticker']]['cost'] -= cost_of_sold
                    temp_portfolio[row['ticker']]['qty'] -= row['quantity']
        
        wins = [p for p in pnl if p > 0]
        losses = [p for p in pnl if p < 0]
        win_ratio = (len(wins) / len(pnl) * 100) if pnl else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        unrealized_pnl = 0
        final_portfolio = []
        last_date = self.portfolio_value_history[-1]['date']
        last_date_dt = datetime.strptime(last_date, '%Y-%m-%d')

        for ticker, pos in self.portfolio.items():
            last_price = pos['avg_price']
            if ticker in all_indicator_data:
                relevant_prices = all_indicator_data[ticker][all_indicator_data[ticker].index <= last_date_dt]
                if not relevant_prices.empty:
                    last_price = relevant_prices.iloc[-1]['close']
            
            pnl_val = (last_price - pos['avg_price']) * pos['quantity']
            unrealized_pnl += pnl_val
            final_portfolio.append({
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
                "total_trades": len(pnl),
                "win_ratio_pct": win_ratio,
                "avg_win": avg_win,
                "avg_loss": avg_loss
            },
            "pnl_history": self.portfolio_value_history,
            "transactions": self.transactions,
            "final_portfolio": final_portfolio,
            "log": self.log
        }
