# strategy.py

class BaseStrategy:
    """
    Base class for all trading strategies.
    It defines the interface that all strategy classes should follow.
    """
    def get_sell_signals(self, portfolio, all_data):
        """
        Determines which stocks to sell from the current portfolio.

        Args:
            portfolio (dict): The user's current portfolio holdings.
            all_data (dict): A dictionary containing historical and indicator data for all relevant stocks.

        Returns:
            list: A list of dictionaries, where each dictionary represents a stock to sell
                  and includes the ticker and the reason for selling.
        """
        raise NotImplementedError("get_sell_signals() must be implemented by subclass")

    def get_add_on_signals(self, portfolio, all_data):
        """
        Determines if more shares should be bought for existing positions.

        Args:
            portfolio (dict): The user's current portfolio holdings.
            all_data (dict): A dictionary containing historical and indicator data for all relevant stocks.

        Returns:
            list: A list of dictionaries, where each dictionary represents an add-on opportunity,
                  including ticker and details of the trade.
        """
        raise NotImplementedError("get_add_on_signals() must be implemented by subclass")

    def get_buy_signals(self, stock_list, portfolio, all_data, max_positions):
        """
        Determines which new stocks to buy.

        Args:
            stock_list (list): The list of stocks the user is interested in.
            portfolio (dict): The user's current portfolio holdings.
            all_data (dict): A dictionary containing historical and indicator data for all relevant stocks.
            max_positions (int): The maximum number of open positions allowed.

        Returns:
            list: A list of dictionaries, where each dictionary represents a new stock to buy.
        """
        raise NotImplementedError("get_buy_signals() must be implemented by subclass")


class MomentumStrategy(BaseStrategy):
    """
    A momentum-based trading strategy.
    - Sells based on stop-loss or trend weakness (crossing below EMAs).
    - Buys based on strong momentum signals (EMA crossover, RS, ADX, RSI, Volume).
    - Adds to existing positions based on strength (hitting new highs).
    """
    def get_sell_signals(self, portfolio, all_data):
        """Implements the sell logic for the momentum strategy."""
        signals = []
        for ticker, position in portfolio.items():
            if ticker not in all_data or all_data[ticker].empty:
                continue

            stock_data = all_data[ticker]
            latest_data = stock_data.iloc[-1]
            sell_signal = False
            reason = ""

            # Rule 1: 10% Stop-loss for the first tranche
            if position.get('tranche_level') == 1:
                stop_loss_price = position['avg_price'] * 0.90
                if latest_data['Close'] < stop_loss_price:
                    sell_signal = True
                    reason = f"10% stop-loss (Price < ₹{stop_loss_price:.2f})"

            # Rule 2: Price crosses below EMA 11 for positions with 3+ tranches
            if not sell_signal and position.get('tranche_level', 1) >= 3:
                if 'EMA_11' in latest_data and latest_data['Close'] < latest_data['EMA_11']:
                    sell_signal = True
                    reason = "Price crossed below EMA_11"
            
            # Rule 3: Price crosses below EMA 21 (applies to all)
            if not sell_signal:
                if 'EMA_21' in latest_data and latest_data['Close'] < latest_data['EMA_21']:
                    sell_signal = True
                    reason = "Price crossed below EMA_21"

            if sell_signal:
                signals.append({'ticker': ticker, 'reason': reason})
        
        return signals

    def get_add_on_signals(self, portfolio, all_data):
        """Implements the add-on logic for the momentum strategy."""
        signals = []
        for ticker, position in portfolio.items():
            # Check if the next tranche level is defined in the user's settings
            # This check needs to be done in the calling function (app.py/backtester.py)
            # as the strategy class doesn't have direct access to user settings.
            
            if ticker not in all_data or all_data[ticker].empty:
                continue

            stock_data = all_data[ticker]
            latest_data = stock_data.iloc[-1]
            
            # Check for missing columns to prevent errors
            if 'High' not in latest_data or 'Close' not in latest_data or 'Low' not in latest_data or 'DCU_20_20' not in latest_data:
                continue
                
            wick_ratio = (latest_data['High'] - latest_data['Close']) / (latest_data['High'] - latest_data['Low'] + 1e-6)

            # Add-on condition: Stock hits a new 20-period high with low upper wick
            if latest_data['High'] >= latest_data['DCU_20_20'] and wick_ratio < 0.3:
                signals.append({
                    'ticker': ticker,
                    'price': latest_data['Close'],
                    'reason': 'Hit new high (Donchian Channel breakout)'
                })
        
        return signals

    def get_buy_signals(self, stock_list, portfolio, all_data, max_positions):
        """Implements the buy logic for the momentum strategy."""
        signals = []
        
        if len(portfolio) >= max_positions:
            return [] # Return early if max positions are already open

        for ticker in stock_list:
            if len(portfolio) + len(signals) >= max_positions:
                break # Stop looking for new buys if we've hit the limit
            
            if ticker in portfolio:
                continue

            if ticker not in all_data or len(all_data[ticker]) < 2:
                continue

            stock_data = all_data[ticker]
            latest_data = stock_data.iloc[-1]
            previous_data = stock_data.iloc[-2]

            # --- Buy Conditions ---
            # Condition 1: Trend is up (EMA alignment)
            is_ema_crossover = latest_data.get('EMA_11', 0) > latest_data.get('EMA_21', 0)

            # Condition 2: Strong momentum indicators
            is_strong_momentum = (
                latest_data.get('RS', 0) > 1.0 and 
                latest_data.get('ADX_14', 0) > 25 and 
                latest_data.get('RSI_14', 0) > 55
            )

            # Condition 3: Volume confirmation
            is_volume_spike = latest_data.get('Volume', 0) > (1.25 * latest_data.get('Volume_MA10', 0))

            # Condition 4: Price action confirmation
            is_making_higher_close = latest_data.get('Close', 0) > previous_data.get('Close', 0)
            intraday_move_pct = ((latest_data.get('Close', 0) - latest_data.get('Open', 0)) / latest_data.get('Open', 0)) * 100 if latest_data.get('Open', 0) > 0 else 0
            is_not_major_reversal = intraday_move_pct > -1.0

            if (is_ema_crossover and 
                is_strong_momentum and 
                is_volume_spike and
                is_making_higher_close and
                is_not_major_reversal):
                
                signals.append({
                    'ticker': ticker,
                    'price': latest_data['Close'],
                    'reason': 'Strong momentum and volume confirmation'
                })
                
        return signals
