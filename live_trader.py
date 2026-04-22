"""
live_trader.py — Paper Trading Engine for SMC PPO Model

Runs the trained RL agent on live Binance data in paper-trading mode.
No real orders are placed — all positions are tracked virtually.

Usage:
  python live_trader.py                       # paper trading, BTC/USDT 5m
  python live_trader.py --symbol ETH/USDT     # different ticker
  python live_trader.py --balance 5000        # start with $5000 virtual balance
  python live_trader.py --once                # single decision then exit (for cron/testing)
  python live_trader.py --dry-run             # show decisions only, do not persist state

Requirements:
  smc_ppo_model.zip and vec_normalize.pkl must exist (run train_rl.py first).
"""

import os
import json
import time
import glob
import argparse
import importlib
import asyncio
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from telegram import Bot

from rl_env import SMCTradingEnv

load_dotenv()
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN")
try:
    TELEGRAM_CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", "0") or 0)
except (ValueError, TypeError):
    TELEGRAM_CHAT_ID = 0  # Non-numeric value in .env — Telegram alerts disabled

ACTION_NAMES = {0: "HOLD/CLOSE", 1: "BUY LONG", 2: "SELL SHORT"}
PAPER_STATE_FILE = "paper_state.json"
PAPER_TRADES_FILE = "paper_trades.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SMC PPO Paper Trading Engine")
    p.add_argument("--symbol",    type=str,   default="BTC/USDT")
    p.add_argument("--timeframe", type=str,   default="5m")
    p.add_argument("--window",    type=int,   default=64)
    p.add_argument("--balance",   type=float, default=1000.0,
                   help="Starting virtual balance (USD)")
    p.add_argument("--interval",  type=int,   default=300,
                   help="Seconds between decisions (default 300 = 5 min)")
    p.add_argument("--once",      action="store_true",
                   help="Make one decision and exit")
    p.add_argument("--dry-run",   action="store_true",
                   help="Print decisions only — do not save state or trades")
    p.add_argument("--model",     type=str,   default="smc_ppo_model.zip")
    p.add_argument("--vec-norm",  type=str,   default="vec_normalize.pkl")
    return p.parse_args()


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


async def telegram_alert(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        bot = Bot(token=TELEGRAM_TOKEN)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode="Markdown")
    except Exception as e:
        log(f"Telegram failed: {e}")


def send_alert(msg: str):
    try:
        asyncio.run(telegram_alert(msg))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def load_state(initial_balance: float) -> dict:
    if Path(PAPER_STATE_FILE).exists():
        with open(PAPER_STATE_FILE, "r") as f:
            state = json.load(f)
        log(f"Loaded paper state: balance=${state['balance']:.2f}, "
            f"position={state['position']}")
        return state
    return {
        "balance":     initial_balance,
        "peak_balance": initial_balance,
        "position":    0,          # 0=flat, 1=long, -1=short
        "entry_price": 0.0,
        "sl_price":    0.0,
        "tp_price":    0.0,
        "entry_time":  None,
        "trades":      [],
    }


def save_state(state: dict, dry_run: bool):
    if dry_run:
        return
    with open(PAPER_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def append_trade(trade: dict, dry_run: bool):
    trades = []
    if Path(PAPER_TRADES_FILE).exists():
        with open(PAPER_TRADES_FILE, "r") as f:
            try:
                trades = json.load(f)
            except Exception:
                trades = []
    trades.append(trade)
    if not dry_run:
        with open(PAPER_TRADES_FILE, "w") as f:
            json.dump(trades, f, indent=2)
    log(f"Trade logged: {trade}")


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_live_data(symbol: str, timeframe: str, limit: int = 250) -> pd.DataFrame:
    """Fetch the last `limit` candles from Binance (no API key needed for public)."""
    exchange = ccxt.binance({"enableRateLimit": True})
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


# ---------------------------------------------------------------------------
# Observation builder (mirrors training exactly)
# ---------------------------------------------------------------------------

def build_observation(df_features: pd.DataFrame, window: int) -> np.ndarray:
    """
    Build a single observation from the last `window` rows.
    The df_features must already have analyze_chart() applied.
    Returns shape: (window, n_features) float32.
    """
    if len(df_features) < window:
        raise ValueError(f"Need at least {window} candles, got {len(df_features)}")
    return df_features.iloc[-window:].values.astype(np.float32)


def normalize_obs(obs: np.ndarray, vec_env: VecNormalize) -> np.ndarray:
    """Apply VecNormalize obs stats to a raw observation (inference-time)."""
    flat = obs.flatten()
    # VecNormalize works on (n_envs, obs_dim) — add and remove batch dim
    flat_norm = vec_env.normalize_obs(flat[np.newaxis, :])
    return flat_norm[0].reshape(obs.shape)


# ---------------------------------------------------------------------------
# SL/TP check on latest candle
# ---------------------------------------------------------------------------

def check_sl_tp(state: dict, row: pd.Series) -> str | None:
    """
    Returns 'SL' or 'TP' if the current candle triggered a stop/target,
    otherwise None.
    """
    pos = state["position"]
    if pos == 0:
        return None

    high = float(row["High"])
    low  = float(row["Low"])

    if pos == 1:   # long
        if low  <= state["sl_price"]:
            return "SL"
        if high >= state["tp_price"]:
            return "TP"

    elif pos == -1:  # short
        if high >= state["sl_price"]:
            return "SL"
        if low  <= state["tp_price"]:
            return "TP"

    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Validate model files ---
    for f in (args.model, args.vec_norm):
        if not Path(f).exists():
            print(f"ERROR: '{f}' not found. Run train_rl.py first.")
            return

    # --- Load latest SMC strategy ---
    files = glob.glob("python_strategy_v*.py")
    if not files:
        print("ERROR: No python_strategy_v*.py strategy file found!")
        return
    latest_v       = sorted([int(f.split("_v")[1].split(".")[0]) for f in files])[-1]
    strategy_mod   = importlib.import_module(f"python_strategy_v{latest_v}")
    log(f"Strategy: python_strategy_v{latest_v}")

    # --- Load trained PPO model ---
    log(f"Loading model: {args.model}")
    model = PPO.load(args.model)

    # --- Load VecNormalize stats for correct observation scaling ---
    # We need a dummy env with the right obs shape to load VecNormalize.
    # Fetch a small slice of data to bootstrap the env shape.
    log("Bootstrapping feature dimensions from live data...")
    bootstrap_df = fetch_live_data(args.symbol, args.timeframe, limit=150)
    bootstrap_df = strategy_mod.analyze_chart(bootstrap_df.copy())
    bootstrap_df.dropna(inplace=True)

    dummy_env    = DummyVecEnv([lambda: SMCTradingEnv(bootstrap_df, window_size=args.window)])
    vec_env      = VecNormalize.load(args.vec_norm, dummy_env)
    vec_env.training    = False
    vec_env.norm_reward = False
    log("VecNormalize loaded — obs normalization active.")

    # --- Load / initialise paper trading state ---
    state = load_state(args.balance)

    log(f"Starting paper trading loop: {args.symbol} @ {args.timeframe}")
    log(f"  Balance: ${state['balance']:.2f} | Interval: {args.interval}s")
    log("  Press Ctrl+C to stop.\n")

    iteration = 0
    while True:
        iteration += 1
        log(f"--- Iteration {iteration} ---")

        try:
            # 1. Fetch live data
            df_live = fetch_live_data(args.symbol, args.timeframe, limit=200)
            log(f"Fetched {len(df_live)} candles. Latest close: {df_live['Close'].iloc[-1]:.2f}")

            # Use the second-to-last candle as "current" (last candle is still forming)
            df_confirmed = df_live.iloc[:-1].copy()

            # 2. Apply SMC indicators
            df_feat = strategy_mod.analyze_chart(df_confirmed.copy())
            df_feat.dropna(inplace=True)

            if len(df_feat) < args.window:
                log(f"Not enough confirmed candles ({len(df_feat)}) — skipping.")
                if args.once:
                    break
                time.sleep(args.interval)
                continue

            # 3. Check SL/TP on the latest confirmed candle
            latest_row  = df_feat.iloc[-1]
            current_price = float(latest_row["Close"])
            sl_tp_result  = check_sl_tp(state, latest_row)

            if sl_tp_result and state["position"] != 0:
                fill_price = state["sl_price"] if sl_tp_result == "SL" else state["tp_price"]
                pos        = state["position"]

                if pos == 1:
                    pnl_pct = (fill_price - state["entry_price"]) / state["entry_price"] * 100
                else:
                    pnl_pct = (state["entry_price"] - fill_price) / state["entry_price"] * 100

                state["balance"] *= (1 + pnl_pct / 100 - 0.001)  # apply fee
                state["peak_balance"] = max(state["peak_balance"], state["balance"])

                direction  = "LONG" if pos == 1 else "SHORT"
                trade_log  = {
                    "time":       datetime.now(timezone.utc).isoformat(),
                    "symbol":     args.symbol,
                    "direction":  direction,
                    "entry":      state["entry_price"],
                    "exit":       fill_price,
                    "exit_reason": sl_tp_result,
                    "pnl_pct":    round(pnl_pct, 4),
                    "balance":    round(state["balance"], 4),
                }
                append_trade(trade_log, args.dry_run)

                emoji = "✅" if pnl_pct > 0 else "🔴"
                msg = (
                    f"{emoji} *Paper Trade Closed — {sl_tp_result}*\n"
                    f"Symbol: {args.symbol}  |  Direction: {direction}\n"
                    f"Entry: {state['entry_price']:.2f}  |  Exit: {fill_price:.2f}\n"
                    f"PnL: {pnl_pct:+.2f}%  |  Balance: ${state['balance']:.2f}"
                )
                log(msg.replace("*", "").replace("\n", "  "))
                send_alert(msg)

                state["position"]    = 0
                state["entry_price"] = 0.0
                state["sl_price"]    = 0.0
                state["tp_price"]    = 0.0
                state["entry_time"]  = None

            # 4. Build observation and get model decision
            # MUST match SMCTradingEnv._build_obs() exactly:
            #   obs = [market_window (window, n_market)] + [state_row (window, 3)]
            #   state_row = [position/1, unrealized_pnl, steps_in_trade/window]
            temp_env = SMCTradingEnv(df_feat, window_size=args.window)
            market_window = temp_env._market_features[-args.window:].copy()  # (window, n_market)

            # Compute state features (same as env._build_obs)
            pos_norm    = float(state["position"])           # -1, 0, or 1
            if state["position"] != 0 and state["entry_price"] > 0:
                if state["position"] == 1:
                    unrealized = (current_price - state["entry_price"]) / state["entry_price"]
                else:
                    unrealized = (state["entry_price"] - current_price) / state["entry_price"]
                unrealized = float(np.clip(unrealized, -0.5, 0.5))
            else:
                unrealized = 0.0
            # steps_in_trade: approximate from entry_time
            steps_in_trade_norm = 0.0

            state_row  = np.array([pos_norm, unrealized, steps_in_trade_norm], dtype=np.float32)
            state_rows = np.tile(state_row, (args.window, 1))          # (window, 3)
            raw_obs    = np.concatenate([market_window, state_rows], axis=1)  # (window, n_market+3)

            # Flatten → normalize → reshape for model.predict
            raw_obs_flat = raw_obs[np.newaxis, :]               # (1, window, n_market+3)
            norm_obs     = vec_env.normalize_obs(raw_obs_flat)   # VecNormalize handles shape

            action, _ = model.predict(norm_obs, deterministic=True)
            action = int(action[0])
            action_name = ACTION_NAMES[action]
            log(f"Model decision: {action_name}  |  Price: {current_price:.2f}  |  "
                f"Position: {state['position']}  |  Balance: ${state['balance']:.2f}")

            # 5. Execute paper action
            if state["position"] != 0:
                # While in trade, only action 0 (model choosing to close manually)
                # triggers an early exit; SL/TP handled above.
                if action == 0 and state["position"] != 0:
                    pos = state["position"]
                    if pos == 1:
                        pnl_pct = (current_price - state["entry_price"]) / state["entry_price"] * 100
                    else:
                        pnl_pct = (state["entry_price"] - current_price) / state["entry_price"] * 100

                    state["balance"] *= (1 + pnl_pct / 100 - 0.001)
                    state["peak_balance"] = max(state["peak_balance"], state["balance"])

                    direction = "LONG" if pos == 1 else "SHORT"
                    trade_log = {
                        "time":       datetime.now(timezone.utc).isoformat(),
                        "symbol":     args.symbol,
                        "direction":  direction,
                        "entry":      state["entry_price"],
                        "exit":       current_price,
                        "exit_reason": "MODEL_CLOSE",
                        "pnl_pct":    round(pnl_pct, 4),
                        "balance":    round(state["balance"], 4),
                    }
                    append_trade(trade_log, args.dry_run)
                    emoji = "✅" if pnl_pct > 0 else "🔴"
                    msg = (
                        f"{emoji} *Paper Trade Closed — Model Decision*\n"
                        f"{direction}  |  Entry: {state['entry_price']:.2f}  → "
                        f"Exit: {current_price:.2f}\n"
                        f"PnL: {pnl_pct:+.2f}%  |  Balance: ${state['balance']:.2f}"
                    )
                    log(msg.replace("*", "").replace("\n", "  "))
                    send_alert(msg)
                    state["position"] = 0
                    state["entry_time"] = None

            else:
                # Flat — open new position if model says Long or Short
                if action in (1, 2) and "ATR" in df_feat.columns:
                    atr        = float(latest_row.get("ATR", latest_row.get("atr", current_price * 0.005)))
                    direction  = "LONG" if action == 1 else "SHORT"
                    sl_mult    = 1.5
                    tp_mult    = 3.0

                    state["position"]    = 1 if action == 1 else -1
                    state["entry_price"] = current_price
                    state["entry_time"]  = datetime.now(timezone.utc).isoformat()

                    if action == 1:
                        state["sl_price"] = current_price - atr * sl_mult
                        state["tp_price"] = current_price + atr * tp_mult
                    else:
                        state["sl_price"] = current_price + atr * sl_mult
                        state["tp_price"] = current_price - atr * tp_mult

                    msg = (
                        f"📊 *Paper Trade Opened — {direction}*\n"
                        f"Symbol: {args.symbol}\n"
                        f"Entry: {current_price:.2f}\n"
                        f"SL:    {state['sl_price']:.2f}  "
                        f"TP: {state['tp_price']:.2f}\n"
                        f"ATR:   {atr:.2f}  |  Balance: ${state['balance']:.2f}"
                    )
                    log(msg.replace("*", "").replace("\n", "  "))
                    send_alert(msg)

            # 6. Save state
            save_state(state, args.dry_run)

        except KeyboardInterrupt:
            log("Interrupted by user. Exiting.")
            break
        except Exception as e:
            log(f"ERROR in iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()

        if args.once:
            log("--once flag set. Exiting after first decision.")
            break

        log(f"Sleeping {args.interval}s until next candle...\n")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
