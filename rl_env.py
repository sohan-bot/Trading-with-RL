"""
rl_env.py — SMC Signal-Filter Trading Environment (v4)

ROOT CAUSE OF PREVIOUS -80% PnL:
  Model entered on 80% of NO-SIGNAL bars. SMC signals fire only ~0.4% of the time.
  The model learned spurious patterns (noise), not real edge.

REDESIGN:
  The model's job is NOT to decide when to enter from scratch.
  Its job is to FILTER which SMC signals to take vs. skip.

  Each episode starts at a random SMC signal bar.
  Agent sees 64 candles of context leading up to the signal.
  Agent makes ONE decision: 1=Take the signal, 0=Skip it.
  If taken: fixed SL/TP manages the trade, reward = realized PnL fraction.
  If skipped: reward = 0, episode ends immediately.

  Between signals: agent always holds (no choice).

This reduces the problem from "when to trade across 200K bars"
to "is THIS specific SMC setup worth trading?" — tractable for RL.

Reward:
  Take + Win  (TP hit) : +tp/sl ratio  (e.g. +2.0 for 3:1.5 params)
  Take + Loss (SL hit) : -1.0
  Skip                 : 0.0

With this reward, the breakeven win rate for the agent = sl/(sl+tp) = 1.5/(1.5+3) = 33%.
The historical strategy wins ~44%, so there IS edge to exploit.
The model learns context features that predict which signals will win.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class SMCTradingEnv(gym.Env):
    """
    Signal-filter RL environment.
    
    Episode structure:
      1. Reset → pick random bar where SMC generated a signal
      2. Build 64-bar observation window ending at that signal bar
      3. Agent acts: 0=Skip, 1=Enter
         (In training, action 2 also treated as Enter for the signal direction)
      4a. Skip: reward=0, episode done
      4b. Enter: run through bars until SL or TP hit → reward = pnl fraction
      5. Done
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 64,
        initial_balance: float = 1000.0,
        sl_atr_multi: float = 1.5,
        tp_atr_multi: float = 3.0,
        transaction_cost_pct: float = 0.001,
        max_bars_in_trade: int = 200,    # force-close after this many bars
    ):
        super().__init__()

        self.df             = df.reset_index(drop=True)
        self.raw_df_index   = df.index if hasattr(df, 'index') else None
        self.window_size    = window_size
        self.initial_balance = initial_balance
        self.sl_atr_multi   = sl_atr_multi
        self.tp_atr_multi   = tp_atr_multi
        self.transaction_cost = transaction_cost_pct
        self.max_bars_in_trade = max_bars_in_trade

        # ---- Locate all signal bars ----
        if 'signal' in self.df.columns:
            sig = self.df['signal'].fillna(0)
            self._signal_indices = sig[
                (sig != 0) & (sig.index >= window_size)
            ].index.tolist()
        else:
            self._signal_indices = []

        if len(self._signal_indices) < 5:
            # Fallback: use every 50th bar so training can still run
            self._signal_indices = list(range(window_size, len(self.df) - max_bars_in_trade - 1, 50))

        # ---- Build feature matrix ----
        self._market_features = self._build_market_features()
        n_market = self._market_features.shape[1]

        # Obs: market window + 3 state features [position, unrealized_pnl, steps_in_trade/64]
        self._n_state = 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size, n_market + self._n_state),
            dtype=np.float32,
        )

        # Actions: 0=Hold/Skip  1=Buy Long  2=Sell Short
        # During signal-filter episodes: 1/2 both mean "take the signal in its direction"
        self.action_space = spaces.Discrete(3)

        # Episode state
        self._reset_state()

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _build_market_features(self) -> np.ndarray:
        proc = self.df.copy()

        # Timestamp features (session awareness)
        idx = None
        if isinstance(self.raw_df_index, pd.DatetimeIndex):
            idx = self.raw_df_index
        elif 'timestamp' in proc.columns:
            idx = pd.to_datetime(proc['timestamp'])

        if idx is not None:
            hours = np.array(idx.hour, dtype=float)
            dows  = np.array(idx.dayofweek, dtype=float)
            proc['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
            proc['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)
            proc['dow_sin']  = np.sin(2 * np.pi * dows  / 7.0)
            proc['dow_cos']  = np.cos(2 * np.pi * dows  / 7.0)
        else:
            proc['hour_sin'] = proc['hour_cos'] = proc['dow_sin'] = proc['dow_cos'] = 0.0

        # Relativize price-like columns (stationarity)
        close_vals = self.df['Close'].values + 1e-8
        price_keys = ('Open', 'High', 'Low', 'Close', 'SMA', 'ATR', 'atr',
                      'fvg_bull', 'fvg_bear', 'fvg_', 'level', 'EMA', 'bb_')
        for col in list(proc.columns):
            if proc[col].dtype not in (np.float64, np.float32, np.int64, np.int32):
                continue
            if any(k in col for k in price_keys):
                proc[col] = (proc[col].values / close_vals) - 1.0

        # Volume: log + z-score
        if 'Volume' in proc.columns:
            v = np.log1p(np.abs(proc['Volume'].values))
            proc['Volume'] = (v - v.mean()) / (v.std() + 1e-8)

        arr = proc.select_dtypes(include=[np.number]).fillna(0).values
        return arr.astype(np.float32)

    # ------------------------------------------------------------------
    # Episode state
    # ------------------------------------------------------------------

    def _reset_state(self):
        self.balance        = float(self.initial_balance)
        self.position       = 0      # 0=flat, 1=long, -1=short
        self.entry_price    = 0.0
        self.sl_price       = 0.0
        self.tp_price       = 0.0
        self.signal_step    = 0      # the signal bar index
        self.trade_step     = 0      # current bar (advances during trade)
        self.steps_in_trade = 0
        self._decided       = False  # whether agent acted on the signal yet
        self._signal_dir    = 0      # +1 or -1 direction of current signal

    def _get_atr(self, idx: int) -> float:
        row = self.df.loc[idx]
        for col in ('ATR', 'atr'):
            if col in self.df.columns:
                v = row.get(col, 0)
                if v and not np.isnan(v) and v > 0:
                    return float(v)
        return float(row['Close']) * 0.005

    def _build_obs(self, step_idx: int) -> np.ndarray:
        """(window, n_market + 3) observation at step_idx."""
        lo = step_idx - self.window_size
        hi = step_idx
        market_w = self._market_features[lo:hi].copy()

        # State features broadcast across all rows
        if self.position != 0 and self.entry_price > 0:
            cur_price = float(self.df.loc[step_idx, 'Close'])
            if self.position == 1:
                unreal = (cur_price - self.entry_price) / self.entry_price
            else:
                unreal = (self.entry_price - cur_price) / self.entry_price
            unreal = float(np.clip(unreal, -0.5, 0.5))
        else:
            unreal = 0.0

        state_row  = np.array([
            float(self.position),
            unreal,
            min(self.steps_in_trade / self.window_size, 1.0),
        ], dtype=np.float32)
        state_rows = np.tile(state_row, (self.window_size, 1))
        return np.concatenate([market_w, state_rows], axis=1).astype(np.float32)

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()

        # Pick a random signal bar with enough future room for the trade
        valid = [i for i in self._signal_indices
                 if i >= self.window_size and i + self.max_bars_in_trade < len(self.df)]
        if not valid:
            # Fallback if no valid signals found
            valid = list(range(self.window_size, max(self.window_size+1, len(self.df)-self.max_bars_in_trade-1)))

        self.signal_step  = int(np.random.choice(valid))
        self.trade_step   = self.signal_step
        self._signal_dir  = int(self.df.loc[self.signal_step, 'signal']) if 'signal' in self.df.columns else 1
        if self._signal_dir == 0:
            self._signal_dir = 1  # fallback direction

        return self._build_obs(self.signal_step), {}

    def step(self, action: int):
        action = int(action)
        reward = 0.0
        info   = {'balance': self.balance, 'position': self.position}

        # ----------------------------------------------------------------
        # Phase 1: Agent decides at signal bar (first step of episode)
        # ----------------------------------------------------------------
        if not self._decided:
            self._decided = True

            # Smart action masking: actions 1 and 2 both mean "take signal"
            # (respecting direction), action 0 = skip
            take_signal = (action != 0)

            if not take_signal:
                # Skip — episode ends, reward = 0
                return self._build_obs(self.signal_step), 0.0, True, False, info

            # Enter in the direction the SMC signal indicated
            entry_price = float(self.df.loc[self.signal_step, 'Close'])
            atr         = self._get_atr(self.signal_step)
            self.position    = self._signal_dir
            self.entry_price = entry_price

            if self._signal_dir == 1:   # long
                self.sl_price = entry_price - atr * self.sl_atr_multi
                self.tp_price = entry_price + atr * self.tp_atr_multi
            else:                        # short
                self.sl_price = entry_price + atr * self.sl_atr_multi
                self.tp_price = entry_price - atr * self.tp_atr_multi

            self.steps_in_trade = 0
            return self._build_obs(self.signal_step), -self.transaction_cost, False, False, info

        # ----------------------------------------------------------------
        # Phase 2: Trade management — run forward bar-by-bar
        # (action is ignored in this phase, SL/TP manages exit)
        # ----------------------------------------------------------------
        self.trade_step    += 1
        self.steps_in_trade += 1

        if self.trade_step >= len(self.df):
            # Data exhausted — close at market
            cur = float(self.df.loc[self.df.index[-1], 'Close'])
            if self.position == 1:
                pnl = (cur - self.entry_price) / self.entry_price
            else:
                pnl = (self.entry_price - cur) / self.entry_price
            reward        = float(pnl)
            self.position = 0
            done          = True
            info['balance'] = self.balance
            return self._build_obs(min(self.trade_step, len(self.df)-1)), reward, done, False, info

        row  = self.df.loc[self.trade_step]
        high = float(row['High'])
        low  = float(row['Low'])
        done = False

        force_close   = self.steps_in_trade >= self.max_bars_in_trade
        hit_sl, hit_tp = False, False

        if self.position == 1:
            if low <= self.sl_price:
                hit_sl = True
            elif high >= self.tp_price:
                hit_tp = True
        elif self.position == -1:
            if high >= self.sl_price:
                hit_sl = True
            elif low <= self.tp_price:
                hit_tp = True

        if hit_sl:
            # Clean reward: -1 for any SL hit, regardless of distance
            reward        = -1.0
            self.position = 0
            done          = True
        elif hit_tp:
            # Clean reward: +R:R ratio  (e.g. 3/1.5 = +2)
            reward        = self.tp_atr_multi / self.sl_atr_multi
            self.position = 0
            done          = True
        elif force_close:
            # Timeout close at market price
            cur_price = float(row['Close'])
            if self.position == 1:
                pnl = (cur_price - self.entry_price) / self.entry_price
            else:
                pnl = (self.entry_price - cur_price) / self.entry_price
            # Scale timeout reward same as SL/TP for consistent learning
            reward        = float(np.clip(pnl / (self._get_atr(self.signal_step) * self.sl_atr_multi / self.entry_price), -1.0, self.tp_atr_multi / self.sl_atr_multi))
            self.position = 0
            done          = True

        info['balance']  = self.balance
        info['position'] = self.position
        return self._build_obs(self.trade_step), float(reward), done, False, info
