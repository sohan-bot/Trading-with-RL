import importlib.util
from auto_tester import run_backtest, calculate_metrics
import pandas as pd, numpy as np

spec = importlib.util.spec_from_file_location('strat21', 'python_strategy_v21.py')
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

sl_m = mod.PARAMS['sl_atr_multi']
tp_m = mod.PARAMS['tp_atr_multi']
breakeven_wr = sl_m / (sl_m + tp_m) * 100
print(f"PARAMS: SL={sl_m}x, TP={tp_m}x")
print(f"Break-even WR needed: {breakeven_wr:.1f}%  (any WR above this = profitable)")

np.random.seed(42)
n = 1000
close = 40000 + np.cumsum(np.random.randn(n) * 200)
df = pd.DataFrame({
    'Open':   close * 0.999,
    'High':   close * 1.002,
    'Low':    close * 0.998,
    'Close':  close,
    'Volume': np.abs(np.random.randn(n)) * 1e6,
}, index=pd.date_range('2024-01-01', periods=n, freq='5min'))

trades = run_backtest(df, mod)
m = calculate_metrics(trades)
print("\nTest results (fixed SL/TP, no trailing):")
for k, v in m.items():
    print(f"  {k}: {v}")

if trades:
    wins   = [t for t in trades if t['result'] == 'win']
    losses = [t for t in trades if t['result'] == 'loss']
    if wins:
        avg_win = sum(t['pnl_pct'] for t in wins) / len(wins)
        print(f"  avg_win_pct:  {avg_win:.3f}%")
    if losses:
        avg_loss = sum(t['pnl_pct'] for t in losses) / len(losses)
        print(f"  avg_loss_pct: {avg_loss:.3f}%")
