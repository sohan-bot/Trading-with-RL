# Paper Trading Self-Improving System — Setup Guide

## What you get
- `strategy_v1.pine`  — first strategy, paste into TradingView
- `improver.py`       — Flask server: logs trades, reviews every 10, rewrites algo
- `dashboard.html`    — live trade tracker at http://localhost:5000
- Auto-improvement loop: every 10 closed trades Claude analyses + rewrites Pine Script
- Telegram sends you the new script + what changed + why

---

## Step 1 — Install

```bash
pip install flask requests anthropic python-telegram-bot python-dotenv
```

---

## Step 2 — Start the improver server

```bash
python improver.py
```

You'll see:
```
10:00:00  INFO  MFC Improver started on port 5000
10:00:00  INFO  Waiting for TradingView alerts at http://0.0.0.0:5000/trade
```

Open http://localhost:5000 in your browser → dashboard appears (empty until first trade).

---

## Step 3 — Load strategy into TradingView

1. Open TradingView → search BTCUSDT → set chart to **4H timeframe**
2. Click **Pine Editor** (bottom bar)
3. Delete everything → paste contents of `strategy_v1.pine`
4. Click **Save** → give it a name → click **Add to Chart**
5. You'll see EMA lines, coloured background, and buy/sell arrows

---

## Step 4 — Enable Paper Trading

1. Click **Strategy Tester** tab (bottom bar)
2. Click the settings gear → **Broker Emulator**
3. Make sure initial capital is 1000 USDT
4. You'll see the backtest results immediately

---

## Step 5 — Set up the TradingView → Python webhook

TradingView needs to send each trade alert to your Python server.
You need your PC's local IP address for this.

**Find your IP:**
- Windows: open cmd → type `ipconfig` → look for IPv4 (e.g. 192.168.1.5)
- Mac/Linux: `ifconfig` or `ip addr`

**Create TradingView alert:**
1. Right-click on chart → **Add Alert**
2. Condition: select your strategy → **Order fills**
3. Alert message: `{{strategy.order.alert_message}}`
4. Webhook URL: `http://YOUR_IP:5000/trade`
   (e.g. `http://192.168.1.5:5000/trade`)
5. Click **Create**

> Note: TradingView webhooks require a paid plan (Essential or above).
> Free alternative: manually log trades using the /trade endpoint with curl
> or just watch the Strategy Tester and log results manually to trades.json.

---

## Step 6 — Watch it run

Every time TradingView fires an entry or exit alert:
- Your server logs: `Trade #3 closed: win  PnL=+4.20%`
- trades.json updates
- Dashboard at http://localhost:5000 refreshes

After 10 closed trades:
- Claude reviews win rate, R:R, drawdown, patterns
- Writes improved Pine Script
- Telegram sends you the new .pine file + explanation

---

## Step 7 — Update the strategy

When you get a Telegram update:
1. Open the `.pine` file sent to you
2. Go to TradingView Pine Editor
3. Select all (Ctrl+A) → paste new script
4. Save → Add to Chart
5. Re-run the paper trading

The versions are saved locally in `strategy_versions/` so you can always go back.

---

## Manual trade logging (if no TradingView webhook)

You can manually POST a trade result:

```bash
# Log an entry
curl -X POST http://localhost:5000/trade \
  -H "Content-Type: application/json" \
  -d '{"event":"entry","side":"long","price":67420,"sl":65100,"tp":82000,"atr":820,"rsi":52}'

# Log an exit
curl -X POST http://localhost:5000/trade \
  -H "Content-Type: application/json" \
  -d '{"event":"exit","side":"long","price":75000,"reason":"tp"}'
```

---

## How Claude decides what to change

| Win rate  | R:R    | Action                          |
|-----------|--------|---------------------------------|
| ≥ 55%     | ≥ 1.5  | Tune parameters only            |
| 45–55%    | any    | Adjust entry logic + params     |
| < 45%     | any    | Full strategy rewrite           |

Claude also looks at:
- Are short trades losing more than longs? → may disable shorts
- What RSI level correlates with losses? → tighten RSI filter
- Consecutive losses → may add cooldown or trend filter
- Volume at entry → may raise volume threshold

---

## File structure

```
your_folder/
├── strategy_v1.pine          ← seed strategy (paste into TradingView)
├── improver.py               ← Flask server + Claude rewriter
├── dashboard.html            ← web dashboard
├── trades.json               ← auto-created, all trade history
├── .env                      ← your API keys
└── strategy_versions/
    ├── strategy_v1.pine      ← copy of seed
    ├── strategy_v2.pine      ← after first review
    └── strategy_v3.pine      ← after second review ...
```
