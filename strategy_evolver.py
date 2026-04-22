"""
strategy_evolver.py - Deterministic Evolutionary Strategy Optimizer

Replaces Gemini-based rewriting with a local, rule-based evolutionary algorithm:
  1. Diagnose what went wrong in the last backtest (too few trades / high DD / poor R:R)
  2. Generate K targeted parameter mutations
  3. Backtest all K candidates in-process
  4. Keep the one with the highest composite score
  5. Write a new python_strategy_vN.py with those params

No LLM, no API calls, no hallucinated code - parameters only.
The analyze_chart() logic stays fixed; only PARAMS evolves.
"""

import json
import random
import math
from copy import deepcopy
from typing import Callable


# ---------------------------------------------------------------------------
# Parameter search space (min, max, step_type)
# ---------------------------------------------------------------------------
PARAM_SPACE = {
    "atr_period":       {"min": 7,    "max": 28,   "type": "int"},
    "fvg_atr_multi":    {"min": 0.10, "max": 2.50, "type": "float"},
    "sl_atr_multi":     {"min": 0.50, "max": 4.00, "type": "float"},
    "tp_atr_multi":     {"min": 1.00, "max": 9.00, "type": "float"},
    "trail_atr_multi":  {"min": 0.50, "max": 4.00, "type": "float"},
    "sma_fast_period":  {"min": 5,    "max": 50,   "type": "int"},
    "sma_slow_period":  {"min": 20,   "max": 200,  "type": "int"},
    "cmf_period":       {"min": 10,   "max": 50,   "type": "int"},
    "cmf_threshold":    {"min": 0.01, "max": 0.30, "type": "float"},
}

# Target thresholds - what we're aiming for
TARGETS = {
    "min_trades":       50,     # Validation slice must have at least this many trades
    "min_profit_factor": 1.30,
    "max_drawdown":     15.0,   # %
    "min_avg_rr":       2.0,
    "min_win_rate":     0.42,
}


# ---------------------------------------------------------------------------
# Diagnostic engine
# ---------------------------------------------------------------------------

def diagnose(metrics_train: dict, metrics_val: dict) -> dict:
    """
    Look at backtest results and identify what exactly went wrong.
    Returns a dict of boolean flags that guide the mutation strategy.
    """
    tv   = metrics_val.get("total_trades", 0)
    pf_v = metrics_val.get("profit_factor", 0.0)
    pf_t = metrics_train.get("profit_factor", 0.0)
    dd   = metrics_val.get("max_drawdown_pct", 999.0)
    wr   = metrics_val.get("win_rate", 0.0)
    rr   = metrics_val.get("avg_rr", 0.0)

    overfit_ratio = (pf_t / max(pf_v, 0.001)) if pf_v > 0 else 99.0

    problems = {
        "too_few_trades":   tv < TARGETS["min_trades"],
        "too_many_trades":  tv > 400,
        "low_win_rate":     wr < TARGETS["min_win_rate"],
        "high_drawdown":    dd > TARGETS["max_drawdown"],
        "poor_rr":          rr < TARGETS["min_avg_rr"] and tv >= TARGETS["min_trades"],
        "unprofitable":     pf_v < 1.0 and tv >= TARGETS["min_trades"],
        "overfitting":      overfit_ratio > 1.8 and tv >= TARGETS["min_trades"],
        "zero_trades":      tv == 0,
        "n_trades":         tv,
        "val_pf":           pf_v,
        "val_dd":           dd,
    }

    # Build a human-readable explanation
    reasons = []
    if problems["zero_trades"]:      reasons.append("zero trades - filters too strict")
    if problems["too_few_trades"]:   reasons.append(f"only {tv} trades -- need >={TARGETS['min_trades']}")
    if problems["too_many_trades"]:  reasons.append(f"{tv} trades - too noisy, tighten filters")
    if problems["low_win_rate"]:     reasons.append(f"win rate {wr:.1%} - trend filter needs rework")
    if problems["high_drawdown"]:    reasons.append(f"DD {dd:.1f}% - SL too wide")
    if problems["poor_rr"]:          reasons.append(f"R:R {rr:.2f} - TP too close or SL too near")
    if problems["unprofitable"]:     reasons.append(f"PF {pf_v:.2f} - losers > winners")
    if problems["overfitting"]:      reasons.append(f"overfit ratio {overfit_ratio:.1f}x - generalise")
    problems["summary"] = "; ".join(reasons) if reasons else "Parameters are performing well"
    return problems


# ---------------------------------------------------------------------------
# Clamp / validate params
# ---------------------------------------------------------------------------

def _clamp(params: dict) -> dict:
    p = deepcopy(params)
    for key, bounds in PARAM_SPACE.items():
        if key not in p:
            continue
        v = p[key]
        v = max(bounds["min"], min(bounds["max"], v))
        if bounds["type"] == "int":
            v = int(round(v))
        else:
            v = round(float(v), 4)
        p[key] = v

    # Structural constraints
    if p["sma_fast_period"] >= p["sma_slow_period"]:
        p["sma_fast_period"] = max(5, p["sma_slow_period"] // 2)

    if p["tp_atr_multi"] < p["sl_atr_multi"] * 2.0:
        p["tp_atr_multi"] = round(p["sl_atr_multi"] * 2.5, 2)

    if p["trail_atr_multi"] > p["sl_atr_multi"]:
        p["trail_atr_multi"] = round(p["sl_atr_multi"] * 0.9, 2)

    return p


# ---------------------------------------------------------------------------
# Mutation strategies
# ---------------------------------------------------------------------------

def _mutate(params: dict, problems: dict, rng: random.Random) -> dict:
    """
    Apply targeted mutations to params based on diagnosed problems.
    Each problem type has specific parameter adjustments - rule-based, no LLM.
    """
    p = deepcopy(params)

    # --- Zero trades / too few: aggressively loosen entry filters ---
    if problems.get("zero_trades") or problems.get("too_few_trades"):
        p["fvg_atr_multi"]  *= rng.uniform(0.40, 0.75)
        p["cmf_threshold"]  *= rng.uniform(0.30, 0.65)
        # Also randomly try wider ATR window
        if rng.random() < 0.4:
            p["atr_period"] = rng.randint(7, 14)

    # --- Too many trades: tighten entry quality ---
    elif problems.get("too_many_trades"):
        p["fvg_atr_multi"]  *= rng.uniform(1.20, 1.60)
        p["cmf_threshold"]  *= rng.uniform(1.30, 2.00)

    # --- Low win rate: fix trend filter ---
    if problems.get("low_win_rate"):
        # Try both wider and narrower trend windows randomly
        fast = rng.randint(8, 30)
        slow = rng.randint(fast * 2, min(fast * 6, 200))
        p["sma_fast_period"] = fast
        p["sma_slow_period"] = slow

    # --- High drawdown: tighten SL ---
    if problems.get("high_drawdown"):
        p["sl_atr_multi"]   *= rng.uniform(0.60, 0.85)
        p["trail_atr_multi"] = p["sl_atr_multi"] * rng.uniform(0.7, 0.9)

    # --- Poor R:R: push TP further ---
    if problems.get("poor_rr"):
        p["tp_atr_multi"] *= rng.uniform(1.25, 1.70)
        p["sl_atr_multi"] *= rng.uniform(0.85, 1.00)   # keep or tighten SL

    # --- Unprofitable (with enough trades): balanced adjustment ---
    if problems.get("unprofitable") and not problems.get("too_few_trades"):
        p["tp_atr_multi"]  *= rng.uniform(1.10, 1.40)
        p["sl_atr_multi"]  *= rng.uniform(0.80, 0.95)
        p["fvg_atr_multi"] *= rng.uniform(0.85, 1.15)  # small noise

    # --- Overfitting: loosen/generalise all filters ---
    if problems.get("overfitting"):
        p["fvg_atr_multi"]  *= rng.uniform(0.75, 1.00)
        p["cmf_threshold"]  *= rng.uniform(0.60, 0.90)
        # Small noise on everything else to escape local optimum
        for key in ("sl_atr_multi", "tp_atr_multi", "trail_atr_multi"):
            p[key] *= rng.uniform(0.90, 1.10)

    # --- Always add small stochastic noise to explore neighbourhood ---
    noise_keys = ["fvg_atr_multi", "sl_atr_multi", "tp_atr_multi", "cmf_threshold"]
    for key in noise_keys:
        if rng.random() < 0.50:   # 50% chance of perturbing each param
            p[key] *= rng.uniform(0.88, 1.12)

    # --- Occasional random jump to escape local optima (10% chance) ---
    if rng.random() < 0.10:
        jump_key = rng.choice(list(PARAM_SPACE.keys()))
        bounds   = PARAM_SPACE[jump_key]
        if bounds["type"] == "int":
            p[jump_key] = rng.randint(bounds["min"], bounds["max"])
        else:
            p[jump_key] = rng.uniform(bounds["min"], bounds["max"])

    return _clamp(p)


def crossover(params_a: dict, params_b: dict, rng: random.Random) -> dict:
    """
    Blend two parameter sets (parents) to create an offspring.
    For each parameter, randomly take from A or blend A+B.
    """
    child = {}
    for key in params_a:
        if key not in params_b:
            child[key] = params_a[key]
            continue
        r = rng.random()
        if r < 0.45:
            child[key] = params_a[key]
        elif r < 0.90:
            child[key] = params_b[key]
        else:
            # Linear blend
            alpha = rng.random()
            child[key] = params_a[key] * alpha + params_b[key] * (1 - alpha)
    return _clamp(child)


# ---------------------------------------------------------------------------
# Composite scoring function
# ---------------------------------------------------------------------------

def score(metrics: dict) -> float:
    """
    A single number that ranks a candidate.
    Higher is better. Returns -inf if disqualified.
    """
    tv = metrics.get("total_trades", 0)
    if tv < TARGETS["min_trades"]:
        return -1e9   # disqualify: not enough trades to be statistically meaningful

    pf  = metrics.get("profit_factor",    0.0)
    dd  = metrics.get("max_drawdown_pct", 999.0)
    wr  = metrics.get("win_rate",         0.0)
    rr  = min(metrics.get("avg_rr",       0.0), 5.0)  # cap at 5 to avoid cherry-picking

    # Penalise hard if drawdown exceeds target
    dd_penalty = max(0.0, (dd - TARGETS["max_drawdown"]) * 0.05)

    # Reward: PF × win_rate × R:R × log(trade_count) − drawdown_penalty
    s = (pf * 0.5 + wr * 0.25 + rr * 0.1 + math.log(tv + 1) * 0.05) - dd_penalty
    return s


# ---------------------------------------------------------------------------
# Tournament - test K mutations, return best params + metrics
# ---------------------------------------------------------------------------

def tournament(
    current_params: dict,
    problems: dict,
    history: list,
    df_train,
    df_val,
    strategy_template_fn: Callable,   # fn(params) → strategy module with analyze_chart
    run_backtest_fn: Callable,
    calc_metrics_fn: Callable,
    k: int = 8,
    seed: int = None,
) -> tuple[dict, dict, str]:
    """
    Generate k candidate mutations, backtest each, return
    (best_params, best_metrics_val, explanation_string).
    """
    rng = random.Random(seed)

    # Pool candidates: mutations of current + crossover with history bests
    candidates = []

    # 1. Pure mutations of current params
    for _ in range(k - 1):
        candidates.append(_mutate(current_params, problems, rng))

    # 2. Crossover with best historical performers (if history has > 1 entry)
    good_history = sorted(
        [h for h in history if h.get("validation_pf", 0) > 0.8],
        key=lambda h: h.get("validation_pf", 0),
        reverse=True,
    )
    if len(good_history) >= 2:
        # Take the two best historical param snapshots and cross them
        h_params_a = good_history[0].get("params", current_params)
        h_params_b = good_history[1].get("params", current_params)
        cx = crossover(h_params_a, h_params_b, rng)
        candidates.append(_mutate(cx, problems, rng))
    elif len(good_history) == 1:
        h_params = good_history[0].get("params", current_params)
        candidates.append(crossover(current_params, h_params, rng))
    else:
        candidates.append(_mutate(current_params, problems, rng))

    # Evaluate each candidate
    best_params  = current_params
    best_score   = score(calc_metrics_fn(run_backtest_fn(df_val, strategy_template_fn(current_params))))
    best_metrics = calc_metrics_fn(run_backtest_fn(df_val, strategy_template_fn(current_params)))

    results = []
    for i, cand_params in enumerate(candidates):
        try:
            mod = strategy_template_fn(cand_params)
            m_val = calc_metrics_fn(run_backtest_fn(df_val, mod))
            s     = score(m_val)
            results.append((s, cand_params, m_val))
        except Exception as e:
            print(f"    [Evolver] Candidate {i+1} failed: {e}")
            continue

    # Sort by score, pick best
    results.sort(key=lambda x: x[0], reverse=True)

    if results and results[0][0] > best_score:
        best_score, best_params, best_metrics = results[0]
        why = _build_explanation(current_params, best_params, problems, best_metrics)
    else:
        why = f"No improvement found over {k} candidates - keeping current params. " + problems["summary"]

    return best_params, best_metrics, why


def _build_explanation(old: dict, new: dict, problems: dict, metrics: dict) -> str:
    """Generate a human-readable diff of what changed and why."""
    changes = []
    for key in old:
        if key not in new:
            continue
        old_v, new_v = old[key], new[key]
        if isinstance(old_v, float):
            if abs(old_v - new_v) / (abs(old_v) + 1e-8) > 0.02:   # >2% change
                direction = "^" if new_v > old_v else "v"
                changes.append(f"  {key}: {old_v:.3f} {direction} {new_v:.3f}")
        elif old_v != new_v:
            changes.append(f"  {key}: {old_v} -> {new_v}")

    reason = problems.get("summary", "general optimisation")
    metrics_str = (
        f"trades={metrics.get('total_trades', 0)}, "
        f"PF={metrics.get('profit_factor', 0):.2f}, "
        f"DD={metrics.get('max_drawdown_pct', 0):.1f}%, "
        f"WR={metrics.get('win_rate', 0):.1%}, "
        f"RR={metrics.get('avg_rr', 0):.2f}"
    )

    return (
        f"Target: {reason}.\n"
        f"Parameter changes:\n" + ("\n".join(changes) if changes else "  (minor noise only)") +
        f"\nResult metrics: {metrics_str}"
    )


# ---------------------------------------------------------------------------
# Strategy code generator - write python_strategy_vN.py from params
# ---------------------------------------------------------------------------

def generate_strategy_code(params: dict, version: int, why: str = "") -> str:
    """
    Generate a complete, valid python_strategy_vN.py from PARAMS dict.
    The analyze_chart logic is canonical and fixed - only PARAMS changes.
    Always uses ASCII-safe content to avoid encoding issues on Windows.
    """
    params_repr = json.dumps(params, indent=4)

    # Strip all non-ASCII characters from why before embedding in source code
    # This prevents cp1252 / UTF-8 decode errors when Python imports the file
    why_safe = why[:200].replace("\n", " ")
    why_safe = "".join(c if ord(c) < 128 else "-" for c in why_safe)

    code = f'''# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pandas_ta as ta

# Auto-generated by StrategyEvolver v{version} (deterministic, no LLM)
# Optimisation note: {why_safe}

PARAMS = {params_repr}


def analyze_chart(df):
    """
    Inputs OHLCV DataFrame and appends Smart Money Concepts signals.
    Entry logic: FVG retest with SMA trend filter and CMF volume confirmation.
    Parameters are evolved automatically - logic is fixed.
    """
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in required):
        raise ValueError(f"DataFrame must contain {{required}}")

    p = PARAMS

    # 1. Trend filter: dual SMA
    df["SMA_FAST"] = ta.sma(df["Close"], length=p["sma_fast_period"])
    df["SMA_SLOW"] = ta.sma(df["Close"], length=p["sma_slow_period"])

    # 2. Volume confirmation: Chaikin Money Flow
    df["CMF"] = ta.cmf(df["High"], df["Low"], df["Close"], df["Volume"],
                       length=p["cmf_period"])

    # 3. Volatility baseline: ATR
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=p["atr_period"])

    # 4. Fair Value Gap detection
    #    Bullish FVG: Low[t] > High[t-2]  - price leaves an upward gap
    #    Bearish FVG: High[t] < Low[t-2]  - price leaves a downward gap
    df["bull_fvg_size_raw"] = df["Low"] - df["High"].shift(2)
    df["bear_fvg_size_raw"] = df["Low"].shift(2) - df["High"]

    min_fvg_size = p["fvg_atr_multi"] * df["ATR"]
    df["bull_fvg_active"] = (
        (df["Low"] > df["High"].shift(2)) &
        (df["bull_fvg_size_raw"] >= min_fvg_size)
    )
    df["bear_fvg_active"] = (
        (df["High"] < df["Low"].shift(2)) &
        (df["bear_fvg_size_raw"] >= min_fvg_size)
    )

    # Capture FVG boundaries at creation for retest checks
    df["fvg_bull_upper"] = df["High"].shift(2)   # upper edge of bullish void
    df["fvg_bull_lower"] = df["Low"]             # lower edge of bullish void
    df["fvg_bear_upper"] = df["High"]            # upper edge of bearish void
    df["fvg_bear_lower"] = df["Low"].shift(2)    # lower edge of bearish void

    df["signal"] = 0

    # 5. Long signal: retest of prior bullish FVG with trend + volume confirmation
    long_cond = (
        df["bull_fvg_active"].shift(1).fillna(False) &                    # FVG formed last bar
        (df["Low"] <= df["fvg_bull_lower"].shift(1)).fillna(False) &       # price retraces into gap
        (df["Close"] > df["fvg_bull_lower"].shift(1)).fillna(False) &      # bounces: close above lower edge
        (df["SMA_FAST"] > df["SMA_SLOW"]).fillna(False) &                  # uptrend confirmed
        (df["CMF"] > p["cmf_threshold"]).fillna(False)                     # buying volume present
    )
    df.loc[long_cond, "signal"] = 1

    # 6. Short signal: retest of prior bearish FVG with trend + volume confirmation
    short_cond = (
        df["bear_fvg_active"].shift(1).fillna(False) &
        (df["High"] >= df["fvg_bear_upper"].shift(1)).fillna(False) &
        (df["Close"] < df["fvg_bear_upper"].shift(1)).fillna(False) &
        (df["SMA_FAST"] < df["SMA_SLOW"]).fillna(False) &
        (df["CMF"] < -p["cmf_threshold"]).fillna(False)
    )
    df.loc[short_cond, "signal"] = -1

    return df
'''
    return code


def write_strategy_file(params: dict, version: int, why: str = "") -> str:
    """Write the generated strategy to disk and return the filename."""
    filename = f"python_strategy_v{version}.py"
    code = generate_strategy_code(params, version, why)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"    [Evolver] Wrote {filename} ({len(code)} chars, UTF-8)")
    return filename


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_params = {
        "atr_period": 14, "fvg_atr_multi": 0.55, "sl_atr_multi": 2.0,
        "tp_atr_multi": 5.0, "trail_atr_multi": 2.0,
        "sma_fast_period": 20, "sma_slow_period": 50,
        "cmf_period": 20, "cmf_threshold": 0.05,
    }
    problems = diagnose(
        {"profit_factor": 2.1, "total_trades": 80},
        {"profit_factor": 0.7, "total_trades": 35, "max_drawdown_pct": 22, "win_rate": 0.38, "avg_rr": 1.4}
    )
    print("Diagnosis:", json.dumps({k: v for k, v in problems.items() if isinstance(v, bool)}, indent=2))
    rng = random.Random(42)
    mutated = _mutate(sample_params, problems, rng)
    print("Mutated params:", json.dumps(mutated, indent=2))
    code = generate_strategy_code(mutated, 99, "self-test")
    print(f"Generated {len(code)} chars of strategy code - OK")
