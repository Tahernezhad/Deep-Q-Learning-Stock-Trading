import numpy as np
import pandas as pd
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import registry, register


class StockTradingEnv(gym.Env):
    """A custom stock trading environment for Gymnasium"""
    metadata = {'render_modes': ['human']}

    def __init__(self, ticker, start_date, end_date, window_size, render_mode=None):
        super(StockTradingEnv, self).__init__()

        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        self.render_mode = render_mode

        # Load data using yfinance
        self.df = self._load_data()
        self.prices = np.asarray(self.df['Close'], dtype=np.float64).reshape(-1)
        self.signal_features = self._process_data()

        self.max_steps = len(self.df) - 1

        # Define action space: 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)

        # Define observation space (the state)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size,), dtype=np.float32
        )

        # Environment state variables
        self.inventory = []
        self.total_profit = 0
        self.current_step = 0

    def _load_data(self):
        print(f"Downloading data for {self.ticker} from {self.start_date} to {self.end_date}...")
        try:
            df = yf.download(
                self.ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                auto_adjust=True
            )
            if df.empty or 'Close' not in df.columns:
                raise ValueError("No adjusted 'Close' data returned.")
            df = df.dropna().copy()
            if len(df) <= self.window_size:
                raise ValueError("Not enough data for the chosen window_size.")
        except Exception as e:
            raise RuntimeError(f"Failed to download data for {self.ticker}: {e}")
        print("Data downloaded successfully.")
        return df

    def _process_data(self):
        # 1) Force a clean 1-D float array of adjusted closes
        prices = np.asarray(self.df['Close'].values, dtype=np.float64).ravel()

        # 2) Build the per-step state as: left-padded raw window -> consecutive diffs (len = window_size)
        signals = []
        for i in range(len(prices)):
            start = max(0, i - self.window_size)
            raw = prices[start:i + 1]
            if raw.size < self.window_size + 1:
                # left-pad with the first price so the window length is fixed
                pad_len = (self.window_size + 1) - raw.size
                raw = np.concatenate([np.full(pad_len, prices[0], dtype=np.float64), raw])
            # state is diffs of that raw window: length == window_size
            state = np.diff(raw).astype(np.float32)
            signals.append(state)

        # Shape: (num_steps, window_size)
        return np.vstack(signals).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.inventory = []
        self.total_profit = 0

        self.current_step = 0

        # Get the first observation (it will be padded with zeros, which is correct)
        observation = self.signal_features[self.current_step]
        info = self._get_info()

        return observation, info

    def step(self, action):
        current_price = float(self.prices[self.current_step])
        reward = 0.0

        if action == 1:  # Buy
            self.inventory.append(current_price)  # allow multiple positions
        elif action == 2:  # Sell
            if self.inventory:
                bought_price = self.inventory.pop(0)
                profit = current_price - bought_price
                reward = profit
                self.total_profit += profit

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        next_observation = self.signal_features[self.current_step]
        truncated = False
        info = self._get_info()
        if self.render_mode == 'human' and self.current_step % 200 == 0:
            print(self._render_human())
        return next_observation, reward, terminated, truncated, info

    def _get_info(self):
        return {
            'total_profit': float(self.total_profit),
            'current_price': float(self.prices[self.current_step]),
            'inventory_size': len(self.inventory),
        }

    def _render_human(self):
        price = float(self.prices[self.current_step])
        return (f"Step: {self.current_step}, Price: {price:.2f}, "
                f"Profit: {float(self.total_profit):.2f}, Inventory: {len(self.inventory)}")

    def render(self):
        if self.render_mode == 'human':
            print(self._render_human())

    def close(self):
        pass

if 'StockTrading-v0' not in registry:
    register(
        id='StockTrading-v0',
        entry_point='stock_env:StockTradingEnv',  # <-- point to the module
    )