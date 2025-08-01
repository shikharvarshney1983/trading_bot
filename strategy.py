# strategy.py

class BaseStrategy:
    """
    Base class for all trading strategies.
    It defines the interface that all strategy classes should follow.
    """
    def get_sell_signals(self, portfolio, all_data):
        raise NotImplementedError("get_sell_signals() must be implemented by subclass")

    def get_add_on_signals(self, portfolio, all_data):
        raise NotImplementedError("get_add_on_signals() must be implemented by subclass")

    def get_buy_signals(self, stock_list, portfolio, all_data, max_positions):
        raise NotImplementedError("get_buy_signals() must be implemented by subclass")


class MomentumStrategy(BaseStrategy):
    """
    A momentum-based trading strategy that is now configurable.
    """
    def __init__(self, params=None):
        """
        Initializes the strategy with a set of parameters.
        
        Args:
            params (dict, optional): A dictionary of strategy parameters. 
                                     Defaults to a standard set if None.
        """
        if params is None:
            # Default parameters if none are provided
            self.params = {
                "stop_loss_pct": 0.10,      # 10% stop-loss
                "rsi_buy_threshold": 55,
                "adx_buy_threshold": 25,
                "volume_spike_factor": 1.25,
                "rs_buy_threshold": 1.0
            }
        else:
            self.params = params

    def get_sell_signals(self, portfolio, all_data):
        """Implements the layered sell logic using configurable parameters."""
        signals = []
        for ticker, position in portfolio.items():
            if ticker not in all_data or all_data[ticker].empty:
                continue

            stock_data = all_data[ticker]
            latest_data = stock_data.iloc[-1]
            sell_signal = False
            reason = ""
            tranche_level = position.get('tranche_level', 1)

            # FIX: Implement layered stop-loss logic
            if tranche_level <= 2:
                # For Tranches 1 & 2: 10% SL or EMA 21, whichever is first
                stop_loss_price = position['avg_price'] * (1 - self.params.get("stop_loss_pct", 0.10))
                if latest_data['Close'] < stop_loss_price:
                    sell_signal = True
                    reason = f"{self.params.get('stop_loss_pct', 0.10)*100}% stop-loss (Price < ₹{stop_loss_price:.2f})"
                elif 'EMA_21' in latest_data and latest_data['Close'] < latest_data['EMA_21']:
                    sell_signal = True
                    reason = "Price crossed below EMA_21"
            else: # tranche_level >= 3
                # For Tranche 3 and above: EMA 11 is the stop-loss
                if 'EMA_11' in latest_data and latest_data['Close'] < latest_data['EMA_11']:
                    sell_signal = True
                    reason = "Price crossed below EMA_11 (Tranche 3+ SL)"

            if sell_signal:
                signals.append({'ticker': ticker, 'reason': reason})
        
        return signals

    def get_add_on_signals(self, portfolio, all_data):
        """Implements the add-on logic."""
        signals = []
        for ticker, position in portfolio.items():
            if ticker not in all_data or all_data[ticker].empty:
                continue

            stock_data = all_data[ticker]
            latest_data = stock_data.iloc[-1]
            
            if 'High' not in latest_data or 'Close' not in latest_data or 'Low' not in latest_data or 'DCU_20_20' not in latest_data:
                continue
                
            wick_ratio = (latest_data['High'] - latest_data['Close']) / (latest_data['High'] - latest_data['Low'] + 1e-6)

            if latest_data['High'] >= latest_data['DCU_20_20'] and wick_ratio < 0.3:
                signals.append({
                    'ticker': ticker,
                    'price': latest_data['Close'],
                    'reason': 'Hit new high (Donchian Channel breakout)'
                })
        
        return signals

    def get_buy_signals(self, stock_list, portfolio, all_data, max_positions):
        """Implements the buy logic using configurable parameters."""
        signals = []
        
        if len(portfolio) >= max_positions:
            return []

        for ticker in stock_list:
            if len(portfolio) + len(signals) >= max_positions:
                break
            
            if ticker in portfolio or ticker not in all_data or len(all_data[ticker]) < 6: # Need at least 6 periods for EMA 40 check
                continue

            stock_data = all_data[ticker]
            latest_data = stock_data.iloc[-1]

            # --- Buy Conditions using configurable parameters ---
            is_trend_aligned = (
                latest_data.get('Close', 0) > latest_data.get('EMA_11', 0) and
                latest_data.get('EMA_11', 0) > latest_data.get('EMA_21', 0) > latest_data.get('EMA_40', 0)
            )

            is_strong_momentum = (
                latest_data.get('RS', 0) > self.params.get("rs_buy_threshold", 1.0) and 
                latest_data.get('ADX_14', 0) > self.params.get("adx_buy_threshold", 25) and 
                latest_data.get('RSI_14', 0) > self.params.get("rsi_buy_threshold", 55)
            )

            is_volume_spike = latest_data.get('Volume', 0) > (self.params.get("volume_spike_factor", 1.25) * latest_data.get('Volume_MA10', 0))

            # FIX: Add check for upward sloping EMA 40
            is_ema40_sloping_up = stock_data['EMA_40'].iloc[-1] > stock_data['EMA_40'].iloc[-3]

            if is_trend_aligned and is_strong_momentum and is_volume_spike and is_ema40_sloping_up:
                signals.append({
                    'ticker': ticker,
                    'price': latest_data['Close'],
                    'reason': 'Strong momentum and volume confirmation'
                })
                
        return signals
