"""
train_rl.py — SMC Signal-Filter PPO Training (v4)

Philosophy:
  The RL model does NOT decide when to enter from scratch.
  It filters which SMC-generated signals to take vs. skip.
  This is a tractable binary decision problem: the model sees 64-bar context
  at each signal and decides: Enter (1) or Skip (0).

  Reward: +2.0 on TP hit, -1.0 on SL hit, 0.0 on skip.
  Breakeven WR = 33%. Historical strategy WR ~44% → real edge to exploit.

Usage:
  python train_rl.py                          # 500K steps (fast, ~15 min)
  python train_rl.py --timesteps 2000000      # full training (~60 min)
  python train_rl.py --timesteps 30000        # smoke test (~1 min)
"""

import os, sys, glob, argparse, warnings
import importlib, importlib.util
import numpy as np
warnings.filterwarnings('ignore')

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from auto_tester import get_data
from rl_env import SMCTradingEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--symbol",    type=str, default="BTC/USDT")
    p.add_argument("--timeframe", type=str, default="5m")
    p.add_argument("--days-back", type=int, default=750)
    p.add_argument("--window",    type=int, default=64)
    p.add_argument("--no-checkpoint", action="store_true")
    return p.parse_args()


def load_strategy():
    files = glob.glob("python_strategy_v*.py")
    if not files:
        print("ERROR: No python_strategy_v*.py found!"); sys.exit(1)
    v     = sorted([int(f.split("_v")[1].split(".")[0]) for f in files])[-1]
    fname = f"python_strategy_v{v}.py"
    spec  = importlib.util.spec_from_file_location(f"python_strategy_v{v}", fname)
    mod   = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    print(f">>> Strategy: {fname}")
    return mod


class EpisodeLogger(BaseCallback):
    """Logs ep_rew_mean at intervals so we can verify learning."""
    def __init__(self, log_interval=10000):
        super().__init__()
        self.log_interval = log_interval
        self.ep_rewards = []
        self._steps_since_log = 0

    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.ep_rewards.append(info['episode']['r'])
        self._steps_since_log += 1
        if self._steps_since_log >= self.log_interval and self.ep_rewards:
            last_n = self.ep_rewards[-50:]
            mean   = sum(last_n) / len(last_n)
            print(f"  [Step {self.num_timesteps:>7}] ep_rew_mean (last 50 eps): {mean:+.4f}"
                  f"  |  total episodes: {len(self.ep_rewards)}")
            self._steps_since_log = 0
        return True


def validate(model, df_val, vec_norm_path, window):
    """
    Run model on every signal bar in df_val.
    Returns: (filtered_wr, raw_wr, n_filtered_trades, n_total_signals)
    """
    env = SMCTradingEnv(df_val, window_size=window)
    signal_indices = env._signal_indices
    if not signal_indices:
        print("  No signals in validation data — skipping validate.")
        return 0, 0, 0, 0

    # Load VecNormalize for proper scaling
    val_venv  = DummyVecEnv([lambda: SMCTradingEnv(df_val, window_size=window)])
    val_venv  = VecNormalize.load(vec_norm_path, val_venv)
    val_venv.training   = False
    val_venv.norm_reward = False

    # Run each signal bar as its own episode
    filtered_wins = 0; filtered_losses = 0; filtered_skips = 0
    raw_wins = 0; raw_losses = 0

    for sig_idx in signal_indices:
        # --- Filter model decision ---
        ep_env = SMCTradingEnv(df_val, window_size=window)
        ep_env.signal_step = sig_idx
        ep_env.trade_step  = sig_idx
        ep_env._signal_dir = int(df_val.reset_index(drop=True).loc[sig_idx, 'signal']) if 'signal' in df_val.columns else 1
        if ep_env._signal_dir == 0: ep_env._signal_dir = 1

        raw_obs = ep_env._build_obs(sig_idx)[np.newaxis, :]  # (1, window, n_feat)
        norm    = val_venv.normalize_obs(raw_obs)             # VecNormalize handles shape
        obs_in  = norm                                        # (1, window, n_feat)
        action, _ = model.predict(obs_in, deterministic=True)
        action = int(action[0])

        if action == 0:  # Skip
            filtered_skips += 1
            continue

        # Model says take — run to SL/TP
        obs, r, done, _, _ = ep_env.step(1)  # enter
        steps = 0
        total_r = r
        while not done and steps < 300:
            obs, r, done, _, _ = ep_env.step(0)
            total_r += r; steps += 1
        if total_r > 1.0: filtered_wins += 1
        else:              filtered_losses += 1

        # --- Raw (unfiltered) performance ---
        raw_env = SMCTradingEnv(df_val, window_size=window)
        raw_env.signal_step = sig_idx; raw_env.trade_step = sig_idx
        raw_env._signal_dir = ep_env._signal_dir
        obs, r, done, _, _ = raw_env.step(1)
        steps = 0; total_r = r
        while not done and steps < 300:
            obs, r, done, _, _ = raw_env.step(0)
            total_r += r; steps += 1
        if total_r > 1.0: raw_wins += 1
        else: raw_losses += 1

    n_total  = len(signal_indices)
    n_taken  = filtered_wins + filtered_losses
    raw_wr   = raw_wins / max(raw_wins + raw_losses, 1)
    filt_wr  = filtered_wins / max(n_taken, 1)

    print(f"\n  Total signals in val : {n_total}")
    print(f"  Raw strategy         : {raw_wins}W / {raw_losses}L  WR={raw_wr:.1%}")
    print(f"  Model (filtered)     : {filtered_wins}W / {filtered_losses}L  Skipped={filtered_skips}  WR={filt_wr:.1%}")

    return filt_wr, raw_wr, n_taken, n_total


def main():
    args = parse_args()

    print("=" * 60)
    print("  SMC Signal-Filter PPO Training v4")
    print("=" * 60)
    print(f"  Symbol    : {args.symbol} @ {args.timeframe}")
    print(f"  Timesteps : {args.timesteps:,}")
    print(f"  Window    : {args.window} candles per obs")
    print("=" * 60)

    # 1. Data
    print("\n>>> Fetching data...")
    df = get_data(ticker=args.symbol, timeframe=args.timeframe, days_back=args.days_back)

    # 2. Strategy features
    strategy_mod = load_strategy()
    df = strategy_mod.analyze_chart(df); df.dropna(inplace=True)

    sig_count = int((df['signal'] != 0).sum()) if 'signal' in df.columns else 0
    print(f"    Total candles: {len(df)}  |  Signal bars: {sig_count}")

    if sig_count < 10:
        print("ERROR: Too few signals to train. Run auto_tester.py to evolve strategy first.")
        sys.exit(1)

    # 3. Split: train=full, val=last 20%
    val_start = int(len(df) * 0.80)
    df_train  = df
    df_val    = df.iloc[val_start:].copy()
    val_sigs  = int((df_val['signal'] != 0).sum()) if 'signal' in df_val.columns else 0
    print(f"    Train: {len(df_train)} candles  |  Val: {len(df_val)} candles ({val_sigs} signals)")

    # 4. Build envs
    print("\n>>> Building environments...")

    def make_env():
        env = SMCTradingEnv(df_train, window_size=args.window)
        return Monitor(env)

    train_env = DummyVecEnv([make_env])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    obs_shape = train_env.observation_space.shape
    print(f"    Obs shape: {obs_shape}")
    n_signals_train = len(SMCTradingEnv(df_train, window_size=args.window)._signal_indices)
    print(f"    Signal episodes available for training: {n_signals_train}")
    print(f"    Approx episodes per 100K steps: {100000 // max(50, 200)}")   # rough estimate

    # 5. PPO Model
    print("\n>>> Initialising PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,                # silent — we log manually via callback
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,            # encourage exploring both take and skip
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 128], vf=[256, 128])),
    )

    # 6. Callbacks
    callbacks = [EpisodeLogger(log_interval=25000)]
    if not args.no_checkpoint:
        os.makedirs("checkpoints", exist_ok=True)
        callbacks.append(CheckpointCallback(
            save_freq=max(100_000, args.timesteps // 5),
            save_path="checkpoints/",
            name_prefix="smc_ppo",
            save_vecnormalize=True,
        ))

    # 7. Train
    print(f"\n>>> TRAINING {args.timesteps:,} steps...")
    print("    Goal: ep_rew_mean should rise from ~-0.33 (random baseline)")
    print("    toward 0 (break-even) and positive (profitable filtering)\n")
    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    # 8. Save
    vec_norm_path = "vec_normalize.pkl"
    model.save("smc_ppo_model")
    train_env.save(vec_norm_path)
    print(f"\n>>> Saved: smc_ppo_model.zip + {vec_norm_path}")

    # 9. Validation
    print("\n" + "=" * 60)
    print("  VALIDATION — last 20% data")
    print("=" * 60)

    if val_sigs > 0:
        filt_wr, raw_wr, n_taken, n_total = validate(model, df_val, vec_norm_path, args.window)

        breakeven_wr = 1.0 / (1.0 + SMCTradingEnv.__init__.__defaults__[3])  # tp/sl
        # tp=3, sl=1.5 → breakeven = 1/(1+2) = 33.3%

        print()
        if n_taken == 0:
            print("[WARNING] Model skipped ALL signals. Increase --timesteps or ent_coef.")
        elif filt_wr > raw_wr:
            print(f"[GREAT] Model improved WR: {raw_wr:.1%} -> {filt_wr:.1%} (filtering works!)")
            print("  -> Run: python live_trader.py --dry-run")
        elif filt_wr > 0.33:
            print(f"[OK] WR={filt_wr:.1%} > break-even 33%. Model is profitable.")
            print("  -> Run: python live_trader.py --dry-run")
        else:
            cur_ts = args.timesteps
            print(f"[NEEDS MORE TRAINING] WR={filt_wr:.1%} < 33% break-even.")
            print(f"  -> python train_rl.py --timesteps {cur_ts * 3:,}")
    else:
        print("  No signals in val period — increase --days-back or check strategy.")
    print("=" * 60)


if __name__ == "__main__":
    main()
