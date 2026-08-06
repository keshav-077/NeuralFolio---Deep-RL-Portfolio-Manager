# src/environment.py (This is the CORRECT version for 8 features)

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class PortfolioEnv(gym.Env):
    """
    A custom environment for portfolio management that includes macroeconomic data.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df, window_size=30, initial_balance=10000, transaction_cost_pct=0.001):
        super(PortfolioEnv, self).__init__()

        # --- Data Handling ---
        self.df = df
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct

        # --- IMPORTANT: Define asset and macro columns ---
        self.asset_columns = ['AAPL', 'BTC-USD', 'MSFT', 'SPY', 'TLT']
        self.macro_columns = ['Federal Funds Rate', 'CPI', 'VIX']

        self.n_assets = len(self.asset_columns)
        self.n_macro_features = len(self.macro_columns)

        # --- This is the attribute that was missing ---
        self.n_features_per_step = self.n_assets + self.n_macro_features # Should be 8

        # --- Action Space ---
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_assets + 1,), dtype=np.float32
        )

        # --- Observation Space ---
        # Shape: (window_size * total_features) = (30 * 8) = 240
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size * self.n_features_per_step,),
            dtype=np.float32
        )

        # --- Internal State ---
        self._current_step = 0
        self._portfolio_value = 0
        self._weights = np.zeros(self.n_assets + 1)

        # Separate dataframes for prices and macro for easier handling
        self.price_df = self.df[self.asset_columns]
        self.macro_df = self.df[self.macro_columns]

    def reset(self, seed=None):
        super().reset(seed=seed)
        self._current_step = self.window_size
        self._portfolio_value = self.initial_balance

        self._weights = np.zeros(self.n_assets + 1)
        self._weights[-1] = 1.0 # 100% in cash

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        current_portfolio_value = self._portfolio_value

        target_weights = np.exp(action) / np.sum(np.exp(action)) # Softmax

        current_asset_values = self._weights[:-1] * current_portfolio_value
        target_asset_values = target_weights[:-1] * current_portfolio_value
        trades = target_asset_values - current_asset_values
        transaction_costs = np.sum(np.abs(trades)) * self.transaction_cost_pct

        self._balance = current_portfolio_value - transaction_costs
        self._weights = target_weights

        self._current_step += 1

        current_prices = self.price_df.iloc[self._current_step - 1].values
        next_prices = self.price_df.iloc[self._current_step].values

        price_ratio = next_prices / (current_prices + 1e-8) # Add epsilon for safety

        asset_values_after_price_change = (self._weights[:-1] * self._balance) * price_ratio
        new_portfolio_value = np.sum(asset_values_after_price_change) + (self._weights[-1] * self._balance)
        self._portfolio_value = new_portfolio_value

        reward = np.log(new_portfolio_value / (current_portfolio_value + 1e-8)) # Add epsilon

        terminated = bool(self._portfolio_value <= self.initial_balance * 0.5)
        truncated = self._current_step >= len(self.df) - 1

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        """
        Gets the observation for the current time step.
        This includes a window of prices AND a window of macro data.
        """
        price_window = self.price_df.iloc[self._current_step - self.window_size : self._current_step].values
        macro_window = self.macro_df.iloc[self._current_step - self.window_size : self._current_step].values

        # Normalize the price window (relative changes)
        normalized_price_window = price_window / (price_window[0] + 1e-8)

        # Normalize the macro window
        normalized_macro_window = macro_window / (macro_window[0] + 1e-8)

        # Combine the normalized windows
        observation_window = np.concatenate([normalized_price_window, normalized_macro_window], axis=1)

        # Flatten into a 1D vector
        return observation_window.flatten().astype(np.float32)

    def _get_info(self):
        return {
            'step': self._current_step,
            'portfolio_value': self._portfolio_value,
            'weights': self._weights,
        }

    def render(self, mode='human'):
        pass

    def close(self):
        pass