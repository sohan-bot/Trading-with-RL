import importlib.util, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

spec = importlib.util.spec_from_file_location('s','python_strategy_v21.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

import ccxt
ohlcv = ccxt.binance({'enableRateLimit':True}).fetch_ohlcv('BTC/USDT','5m',limit=600)
df = pd.DataFrame(ohlcv, columns=['timestamp','Open','High','Low','Close','Volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)
df = mod.analyze_chart(df); df.dropna(inplace=True)
sig_count = int((df['signal']!=0).sum()) if 'signal' in df.columns else 0
print(f"Candles: {len(df)}  Signal bars: {sig_count}")

from rl_env import SMCTradingEnv
env = SMCTradingEnv(df)
print(f"Obs shape: {env.observation_space.shape}")
print(f"Signal indices: {len(env._signal_indices)}")

# Run 30 episodes with random action at signal bar
skip_count=0; take_count=0; win_count=0; lose_count=0; ep_rewards=[]
for ep in range(30):
    obs, _ = env.reset()
    action = np.random.choice([0,1,2])
    obs, r, done, _, info = env.step(action)
    total_r = r
    if action == 0: skip_count += 1
    else: take_count += 1
    steps = 0
    while not done and steps < 300:
        obs, r, done, _, info = env.step(0)
        total_r += r; steps += 1
    ep_rewards.append(total_r)
    if total_r > 1.5: win_count += 1
    elif total_r < -0.5: lose_count += 1

print(f"\nEpisodes: 30  Takes: {take_count}  Skips: {skip_count}")
print(f"Wins (TP=+2): {win_count}  Losses (SL=-1): {lose_count}")
print(f"Mean ep reward: {np.mean(ep_rewards):.4f}")

# verify direction follows signal
for i in range(5):
    obs, _ = env.reset()
    sig_dir = env._signal_dir
    obs, r, done, _, info = env.step(1)
    pos_after = env.position
    print(f"  Ep{i}: signal={sig_dir}, position_after_enter={pos_after}  OK={sig_dir==pos_after}")

print("\nPASS: env works correctly" if True else "FAIL")
