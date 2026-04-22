"""
auto_tester.py — SMC Strategy Backtester + Evolutionary Self-Improver

No Gemini / LLM dependency. Uses strategy_evolver.py for deterministic
parameter optimisation via tournament selection.

Run:
  python auto_tester.py                 # evolve once, save new strategy version
  python auto_tester.py --generations 5 # run 5 evolution cycles in a row
  python auto_tester.py --candidates 12 # test 12 mutations per generation
"""

import pandas as pd
import ccxt
import importlib
import json
import os
import glob
import argparse
import asyncio
from telegram import Bot
from dotenv import load_dotenv

from strategy_evolver import (
    diagnose,
    tournament,
    write_strategy_file,
    score as evolver_score,
    TARGETS,
)

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
try:
    TELEGRAM_CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", "0") or 0)
except (ValueError, TypeError):
    TELEGRAM_CHAT_ID = 0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_data(ticker="BTC/USDT", timeframe="5m", days_back=750):
    print(f"Fetching ~{days_back}d of {timeframe} {ticker} data from Binance...")
    exchange = ccxt.binance({"enableRateLimit": True})

    now_ms  = exchange.milliseconds()
    since   = now_ms - (days_back * 24 * 60 * 60 * 1000)
    start   = since
    total   = now_ms - start
    all_ohlcv = []

    while since < now_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(ticker, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            pct = min(100, int(((since - start) / total) * 100))
            print(f"\rProgress: {pct}% [{len(all_ohlcv)} candles]", end="", flush=True)
        except Exception as e:
            print(f"\nNetwork error: {e}")
            break

    print("")
    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    print(f"Loaded {len(df)} candles.")
    if df.empty:
        print("ERROR: No data returned. Check network.")
        raise SystemExit(1)
    return df


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def run_backtest(df, strategy_module):
    """
    Run a vectorised trade simulation using the strategy's signals.
    Uses FIXED SL/TP (no trailing stop) — trailing stops were cutting winners
    before TP reached, making win rate and PF appear worse than the true edge.
    """
    try:
        df = strategy_module.analyze_chart(df.copy())
    except Exception as e:
        print(f"  analyze_chart failed: {e}")
        return []

    if "atr" not in df.columns and "ATR" not in df.columns:
        import pandas_ta as ta
        df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
        df["ATR"] = df["ATR"].bfill().ffill()

    atr_col = "ATR" if "ATR" in df.columns else "atr"
    if "signal" not in df.columns:
        df["signal"] = 0

    p = getattr(strategy_module, "PARAMS", {})
    sl_multi = p.get("sl_atr_multi", 1.5)
    tp_multi = p.get("tp_atr_multi", 3.0)

    trades   = []
    position = 0
    entry    = 0.0
    sl = tp  = 0.0

    for _, row in df.iterrows():
        atr = float(row[atr_col]) if not pd.isna(row[atr_col]) else 0.0

        # --- Exits: fixed SL/TP, set at entry and never moved ---
        if position == 1:
            if row["Low"] <= sl:
                pnl = (sl - entry) / entry * 100      # always negative (stop loss)
                trades.append({"result": "loss", "pnl_pct": pnl})
                position = 0
            elif row["High"] >= tp:
                pnl = (tp - entry) / entry * 100      # always positive (take profit)
                trades.append({"result": "win", "pnl_pct": pnl})
                position = 0

        elif position == -1:
            if row["High"] >= sl:
                pnl = (entry - sl) / entry * 100      # always negative (stop loss)
                trades.append({"result": "loss", "pnl_pct": pnl})
                position = 0
            elif row["Low"] <= tp:
                pnl = (entry - tp) / entry * 100      # always positive (take profit)
                trades.append({"result": "win", "pnl_pct": pnl})
                position = 0

        # --- Entries ---
        if position == 0:
            sig = row.get("signal", 0)
            if sig == 1:
                position, entry = 1, row["Close"]
                sl = entry - atr * sl_multi
                tp = entry + atr * tp_multi
            elif sig == -1:
                position, entry = -1, row["Close"]
                sl = entry + atr * sl_multi
                tp = entry - atr * tp_multi

    return trades


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_metrics(trades: list) -> dict:
    if not trades:
        return {"total_trades": 0, "win_rate": 0, "avg_rr": 0,
                "max_drawdown_pct": 0, "profit_factor": 0, "total_pnl_pct": 0}

    wins   = [t for t in trades if t["result"] == "win"]
    losses = [t for t in trades if t["result"] == "loss"]

    win_rate = len(wins) / len(trades)
    avg_win  = sum(t["pnl_pct"] for t in wins)  / max(len(wins), 1)
    avg_loss = sum(t["pnl_pct"] for t in losses) / max(len(losses), 1)
    avg_rr   = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

    equity = 1000.0
    peak   = equity
    max_dd = 0.0
    for t in trades:
        equity *= (1 + t["pnl_pct"] / 100)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd

    gross_wins   = sum(t["pnl_pct"] for t in wins)
    gross_losses = abs(sum(t["pnl_pct"] for t in losses))
    pf = (gross_wins / gross_losses) if gross_losses > 0 else 99.0

    return {
        "total_trades":      len(trades),
        "win_rate":          round(win_rate, 3),
        "avg_rr":            round(avg_rr, 2),
        "max_drawdown_pct":  round(max_dd, 2),
        "profit_factor":     round(pf, 2),
        "total_pnl_pct":     round(sum(t["pnl_pct"] for t in trades), 2),
    }


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

async def _send_telegram(metrics_train: dict, metrics_val: dict, why: str, filename: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        bot  = Bot(token=TELEGRAM_TOKEN)
        text = (
            f"*New Strategy: {filename}*\n\n"
            f"*TRAIN (70%)*\n"
            f"Trades: {metrics_train['total_trades']} | WR: {metrics_train['win_rate']:.1%} | "
            f"PF: {metrics_train['profit_factor']} | DD: {metrics_train['max_drawdown_pct']}%\n\n"
            f"*VALIDATION (30%)*\n"
            f"Trades: {metrics_val['total_trades']} | WR: {metrics_val['win_rate']:.1%} | "
            f"PF: {metrics_val['profit_factor']} | DD: {metrics_val['max_drawdown_pct']}%\n\n"
            f"*What changed:*\n{why[:800]}"
        )
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode="Markdown")
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                await bot.send_document(chat_id=TELEGRAM_CHAT_ID, document=f)
    except Exception as e:
        print(f"Telegram send failed: {e}")


def send_telegram(metrics_train, metrics_val, why, filename):
    try:
        asyncio.run(_send_telegram(metrics_train, metrics_val, why, filename))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helper: build a temporary module-like object from params for the evolver
# ---------------------------------------------------------------------------

def _make_module_from_params(params: dict):
    """
    Dynamically compile a strategy module from params dict.
    Used inside the tournament loop to avoid writing temp files to disk.
    """
    from strategy_evolver import generate_strategy_code
    import types

    code = generate_strategy_code(params, version=0, why="tournament candidate")
    mod  = types.ModuleType("_candidate_strategy")
    exec(compile(code, "<candidate>", "exec"), mod.__dict__)
    return mod


# ---------------------------------------------------------------------------
# Main evolution loop
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SMC Strategy Evolutionary Optimizer")
    p.add_argument("--generations", type=int, default=1,
                   help="How many evolution cycles to run (default: 1)")
    p.add_argument("--candidates",  type=int, default=8,
                   help="Mutations to test per generation (default: 8, more = slower but better)")
    p.add_argument("--symbol",      type=str, default="BTC/USDT")
    p.add_argument("--timeframe",   type=str, default="5m")
    p.add_argument("--days-back",   type=int, default=750)
    p.add_argument("--no-telegram", action="store_true")
    return p.parse_args()


def run_one_generation(df, df_train, df_val, latest_version, history, args):
    """Run a single evolution generation. Returns new version number."""

    module_name     = f"python_strategy_v{latest_version}"
    strategy_module = importlib.import_module(module_name)

    print(f"\n{'='*60}")
    print(f"  GENERATION — based on {module_name}")
    print(f"{'='*60}")

    # --- Current performance baseline ---
    print("\n>>> Running baseline backtest (current strategy)...")
    trades_train = run_backtest(df_train, strategy_module)
    trades_val   = run_backtest(df_val,   strategy_module)
    metrics_train = calculate_metrics(trades_train)
    metrics_val   = calculate_metrics(trades_val)

    print(f"\n  TRAIN  — Trades: {metrics_train['total_trades']:>4}  PF: {metrics_train['profit_factor']:.2f}  "
          f"WR: {metrics_train['win_rate']:.1%}  DD: {metrics_train['max_drawdown_pct']:.1f}%")
    print(f"  VAL    — Trades: {metrics_val['total_trades']:>4}  PF: {metrics_val['profit_factor']:.2f}  "
          f"WR: {metrics_val['win_rate']:.1%}  DD: {metrics_val['max_drawdown_pct']:.1f}%")

    baseline_score = evolver_score(metrics_val)

    # Check if already meeting targets
    already_good = (
        metrics_val.get("profit_factor",    0) >= TARGETS["min_profit_factor"] and
        metrics_val.get("max_drawdown_pct", 999) <= TARGETS["max_drawdown"] and
        metrics_val.get("total_trades",     0) >= TARGETS["min_trades"] and
        metrics_val.get("avg_rr",           0) >= TARGETS["min_avg_rr"]
    )
    if already_good:
        print("\n  ✅ Current strategy already meets all targets! Skipping evolution.")
        return latest_version, metrics_train, metrics_val, "Already meeting targets — no change needed."

    # --- Diagnose problems ---
    problems = diagnose(metrics_train, metrics_val)
    print(f"\n  Diagnosis: {problems['summary']}")

    # --- Current params ---
    current_params = getattr(strategy_module, "PARAMS", {})

    # --- Run tournament ---
    print(f"\n>>> Testing {args.candidates} parameter mutations...")
    best_params, best_metrics_val, why = tournament(
        current_params=current_params,
        problems=problems,
        history=history,
        df_train=df_train,
        df_val=df_val,
        strategy_template_fn=_make_module_from_params,
        run_backtest_fn=run_backtest,
        calc_metrics_fn=calculate_metrics,
        k=args.candidates,
    )

    # Check if tournament improved things
    new_score = evolver_score(best_metrics_val)
    print(f"\n  Best candidate — Trades: {best_metrics_val.get('total_trades',0)}  "
          f"PF: {best_metrics_val.get('profit_factor',0):.2f}  "
          f"DD: {best_metrics_val.get('max_drawdown_pct',0):.1f}%  "
          f"Score: {new_score:.4f}  (was {baseline_score:.4f})")

    if new_score <= baseline_score:
        print("  ⚠  Tournament found no improvement — writing unchanged params as new version anyway.")
        best_params = current_params
        why = "No improvement found after tournament. Keeping current params, incrementing version."

    # --- Write new strategy ---
    new_version  = latest_version + 1
    new_filename = write_strategy_file(best_params, new_version, why)
    print(f"\n  Wrote {new_filename}")

    # --- Import the newly written file ---
    # Must invalidate caches so Python sees the file, then load by path (not by name)
    # to avoid stale cached module from a previous run.
    importlib.invalidate_caches()
    import importlib.util as ilu
    _spec    = ilu.spec_from_file_location(f"python_strategy_v{new_version}", new_filename)
    new_mod  = ilu.module_from_spec(_spec)
    _spec.loader.exec_module(new_mod)

    new_trades_train  = run_backtest(df_train, new_mod)
    new_metrics_train = calculate_metrics(new_trades_train)

    return new_version, new_metrics_train, best_metrics_val, why



def main():
    args = parse_args()

    # --- Discover latest strategy version ---
    files = glob.glob("python_strategy_v*.py")
    versions = []
    for f in files:
        try:
            versions.append(int(f.split("_v")[1].split(".")[0]))
        except Exception:
            pass

    if not versions:
        print("ERROR: No python_strategy_v*.py found!")
        return

    versions.sort()
    latest_version = versions[-1]

    # --- Fetch data (once, shared across all generations) ---
    df = get_data(ticker=args.symbol, timeframe=args.timeframe, days_back=args.days_back)

    split_idx = int(len(df) * 0.70)
    df_train  = df.iloc[:split_idx]
    df_val    = df.iloc[split_idx:]
    print(f"\nWalk-forward split: {len(df_train)} train | {len(df_val)} val candles")

    # --- Load evolution history ---
    history_file = "optimization_history.json"
    if os.path.exists(history_file):
        with open(history_file, encoding="utf-8") as f:
            try:
                history = json.load(f)
            except Exception:
                history = []
    else:
        history = []

    # --- Run N generations ---
    for gen in range(args.generations):
        print(f"\n{'#'*60}")
        print(f"  GENERATION {gen+1} of {args.generations}")
        print(f"{'#'*60}")

        new_version, metrics_train, metrics_val, why = run_one_generation(
            df, df_train, df_val, latest_version, history, args
        )

        # Log to history (with params snapshot for crossover)
        try:
            importlib.invalidate_caches()
            import importlib.util as ilu
            _spec    = ilu.spec_from_file_location(f"python_strategy_v{new_version}", f"python_strategy_v{new_version}.py")
            _new_mod = ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_new_mod)
            params_snapshot = getattr(_new_mod, "PARAMS", {})
        except Exception as e:
            print(f"  Warning: could not read params from new module: {e}")
            params_snapshot = {}

        history.append({
            "version":       new_version,
            "in_sample_pf":  metrics_train.get("profit_factor", 0),
            "validation_pf": metrics_val.get("profit_factor", 0),
            "validation_dd": metrics_val.get("max_drawdown_pct", 0),
            "total_trades":  metrics_train.get("total_trades", 0) + metrics_val.get("total_trades", 0),
            "params":        params_snapshot,
        })

        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # Send Telegram
        new_filename = f"python_strategy_v{new_version}.py"
        if not args.no_telegram:
            print("Sending Telegram notification...")
            send_telegram(metrics_train, metrics_val, why, new_filename)

        # Cleanup old versions (keep last 5)
        versions.append(new_version)
        versions.sort()
        if len(versions) > 5:
            for v in versions[:-5]:
                old = f"python_strategy_v{v}.py"
                if os.path.exists(old):
                    os.remove(old)
                    print(f"Deleted old version: {old}")

        # Update for next generation
        latest_version = new_version
        # Re-import new module for next generation
        importlib.invalidate_caches()

        # Print final summary
        print(f"\n  Final: python_strategy_v{new_version}.py")
        print(f"  Val:   Trades={metrics_val['total_trades']}  PF={metrics_val['profit_factor']:.2f}  "
              f"DD={metrics_val['max_drawdown_pct']:.1f}%  WR={metrics_val['win_rate']:.1%}  "
              f"RR={metrics_val['avg_rr']:.2f}")

        target_met = (
            metrics_val.get("profit_factor", 0) >= TARGETS["min_profit_factor"] and
            metrics_val.get("max_drawdown_pct", 999) <= TARGETS["max_drawdown"] and
            metrics_val.get("total_trades", 0) >= TARGETS["min_trades"]
        )
        if target_met:
            print(f"\n  🎯 TARGET MET! PF≥{TARGETS['min_profit_factor']}, DD≤{TARGETS['max_drawdown']}%")
            print("  → Run: python train_rl.py to retrain the RL model on this strategy")
            break

    print("\nDone.")


if __name__ == "__main__":
    main()
