# Graph Report - .  (2026-04-22)

## Corpus Check
- Corpus is ~17,958 words - fits in a single context window. You may not need a graph.

## Summary
- 116 nodes · 163 edges · 22 communities detected
- Extraction: 83% EXTRACTED · 17% INFERRED · 0% AMBIGUOUS · INFERRED: 27 edges (avg confidence: 0.7)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Live Paper Trading|Live Paper Trading]]
- [[_COMMUNITY_RL Training & Callbacks|RL Training & Callbacks]]
- [[_COMMUNITY_Gymnasium Environment Logic|Gymnasium Environment Logic]]
- [[_COMMUNITY_Strategy Evolution Core|Strategy Evolution Core]]
- [[_COMMUNITY_System Architecture Hub|System Architecture Hub]]
- [[_COMMUNITY_Numba Mocking Layer|Numba Mocking Layer]]
- [[_COMMUNITY_SMC Strategy v21|SMC Strategy v21]]
- [[_COMMUNITY_SMC Strategy v22|SMC Strategy v22]]
- [[_COMMUNITY_SMC Strategy v23|SMC Strategy v23]]
- [[_COMMUNITY_Internal Graphify Utility (9)|Internal Graphify Utility (9)]]
- [[_COMMUNITY_Internal Graphify Utility (10)|Internal Graphify Utility (10)]]
- [[_COMMUNITY_Internal Graphify Utility (11)|Internal Graphify Utility (11)]]
- [[_COMMUNITY_Internal Graphify Utility (12)|Internal Graphify Utility (12)]]
- [[_COMMUNITY_Internal Graphify Utility (13)|Internal Graphify Utility (13)]]
- [[_COMMUNITY_Internal Graphify Utility (14)|Internal Graphify Utility (14)]]
- [[_COMMUNITY_Internal Graphify Utility (15)|Internal Graphify Utility (15)]]
- [[_COMMUNITY_Internal Graphify Utility (16)|Internal Graphify Utility (16)]]
- [[_COMMUNITY_Internal Graphify Utility (17)|Internal Graphify Utility (17)]]
- [[_COMMUNITY_Internal Graphify Utility (18)|Internal Graphify Utility (18)]]
- [[_COMMUNITY_Internal Graphify Utility (19)|Internal Graphify Utility (19)]]
- [[_COMMUNITY_Visual Analytics|Visual Analytics]]
- [[_COMMUNITY_Documentation & Setup|Documentation & Setup]]

## God Nodes (most connected - your core abstractions)
1. `SMCTradingEnv` - 21 edges
2. `main()` - 12 edges
3. `run_one_generation()` - 8 edges
4. `main()` - 8 edges
5. `tournament()` - 7 edges
6. `EpisodeLogger` - 7 edges
7. `validate()` - 7 edges
8. `log()` - 6 edges
9. `Auto Tester` - 6 edges
10. `analyze_chart()` - 5 edges

## Surprising Connections (you probably didn't know these)
- `Auto Tester` --rationale_for--> `Help Documentation`  [INFERRED]
  auto_tester.py → help.txt
- `live_trader.py — Paper Trading Engine for SMC PPO Model  Runs the trained RL age` --uses--> `SMCTradingEnv`  [INFERRED]
  live_trader.py → rl_env.py
- `Fetch the last `limit` candles from Binance (no API key needed for public).` --uses--> `SMCTradingEnv`  [INFERRED]
  live_trader.py → rl_env.py
- `Build a single observation from the last `window` rows.     The df_features must` --uses--> `SMCTradingEnv`  [INFERRED]
  live_trader.py → rl_env.py
- `Apply VecNormalize obs stats to a raw observation (inference-time).` --uses--> `SMCTradingEnv`  [INFERRED]
  live_trader.py → rl_env.py

## Communities

### Community 0 - "Live Paper Trading"
Cohesion: 0.14
Nodes (13): BaseCallback, rl_env.py — SMC Signal-Filter Trading Environment (v4)  ROOT CAUSE OF PREVIOUS -, (window, n_market + 3) observation at step_idx., Signal-filter RL environment.          Episode structure:       1. Reset → pick, SMCTradingEnv, EpisodeLogger, load_strategy(), main() (+5 more)

### Community 1 - "RL Training & Callbacks"
Cohesion: 0.14
Nodes (15): calculate_metrics(), get_data(), main(), _make_module_from_params(), parse_args(), auto_tester.py — SMC Strategy Backtester + Evolutionary Self-Improver  No Gemini, Dynamically compile a strategy module from params dict.     Used inside the tour, Run a single evolution generation. Returns new version number. (+7 more)

### Community 2 - "Gymnasium Environment Logic"
Cohesion: 0.2
Nodes (17): append_trade(), build_observation(), check_sl_tp(), fetch_live_data(), load_state(), log(), main(), normalize_obs() (+9 more)

### Community 3 - "Strategy Evolution Core"
Cohesion: 0.17
Nodes (16): _build_explanation(), _clamp(), crossover(), generate_strategy_code(), _mutate(), strategy_evolver.py - Deterministic Evolutionary Strategy Optimizer  Replaces, Apply targeted mutations to params based on diagnosed problems.     Each proble, Blend two parameter sets (parents) to create an offspring.     For each paramet (+8 more)

### Community 4 - "System Architecture Hub"
Cohesion: 0.24
Nodes (10): Auto Tester, Help Documentation, Live Trader, Numba Hook, SMC Strategy v24, RL Environment, PPO Model, Strategy Evolver (+2 more)

### Community 5 - "Numba Mocking Layer"
Cohesion: 0.7
Nodes (4): guvectorize(), jit(), njit(), vectorize()

### Community 6 - "SMC Strategy v21"
Cohesion: 0.67
Nodes (2): analyze_chart(), Inputs OHLCV DataFrame and appends Smart Money Concepts signals purely natively.

### Community 7 - "SMC Strategy v22"
Cohesion: 0.67
Nodes (2): analyze_chart(), Inputs OHLCV DataFrame and appends Smart Money Concepts signals.     Entry logi

### Community 8 - "SMC Strategy v23"
Cohesion: 0.67
Nodes (2): analyze_chart(), Inputs OHLCV DataFrame and appends Smart Money Concepts signals.     Entry logi

### Community 9 - "Internal Graphify Utility (9)"
Cohesion: 1.0
Nodes (0): 

### Community 10 - "Internal Graphify Utility (10)"
Cohesion: 1.0
Nodes (0): 

### Community 11 - "Internal Graphify Utility (11)"
Cohesion: 1.0
Nodes (0): 

### Community 12 - "Internal Graphify Utility (12)"
Cohesion: 1.0
Nodes (0): 

### Community 13 - "Internal Graphify Utility (13)"
Cohesion: 1.0
Nodes (0): 

### Community 14 - "Internal Graphify Utility (14)"
Cohesion: 1.0
Nodes (0): 

### Community 15 - "Internal Graphify Utility (15)"
Cohesion: 1.0
Nodes (0): 

### Community 16 - "Internal Graphify Utility (16)"
Cohesion: 1.0
Nodes (0): 

### Community 17 - "Internal Graphify Utility (17)"
Cohesion: 1.0
Nodes (0): 

### Community 18 - "Internal Graphify Utility (18)"
Cohesion: 1.0
Nodes (1): SMC Strategy v23

### Community 19 - "Internal Graphify Utility (19)"
Cohesion: 1.0
Nodes (1): SMC Strategy v22

### Community 20 - "Visual Analytics"
Cohesion: 1.0
Nodes (1): Trading Dashboard

### Community 21 - "Documentation & Setup"
Cohesion: 1.0
Nodes (1): Paper Trading Guide

## Knowledge Gaps
- **26 isolated node(s):** `auto_tester.py — SMC Strategy Backtester + Evolutionary Self-Improver  No Gemini`, `Run a vectorised trade simulation using the strategy's signals.     Uses FIXED S`, `Dynamically compile a strategy module from params dict.     Used inside the tour`, `Run a single evolution generation. Returns new version number.`, `Inputs OHLCV DataFrame and appends Smart Money Concepts signals purely natively.` (+21 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Internal Graphify Utility (9)`** (1 nodes): `_ast_extract.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Internal Graphify Utility (10)`** (1 nodes): `_build_graph.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Internal Graphify Utility (11)`** (1 nodes): `_cache_check.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Internal Graphify Utility (12)`** (1 nodes): `_cache_merge.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Internal Graphify Utility (13)`** (1 nodes): `_collect_chunks.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Internal Graphify Utility (14)`** (1 nodes): `_finalize.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Internal Graphify Utility (15)`** (1 nodes): `_label_report.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Internal Graphify Utility (16)`** (1 nodes): `_merge_final.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Internal Graphify Utility (17)`** (1 nodes): `_save_manifest.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Internal Graphify Utility (18)`** (1 nodes): `SMC Strategy v23`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Internal Graphify Utility (19)`** (1 nodes): `SMC Strategy v22`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Visual Analytics`** (1 nodes): `Trading Dashboard`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Documentation & Setup`** (1 nodes): `Paper Trading Guide`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SMCTradingEnv` connect `Live Paper Trading` to `Gymnasium Environment Logic`?**
  _High betweenness centrality (0.193) - this node is a cross-community bridge._
- **Why does `main()` connect `Gymnasium Environment Logic` to `Live Paper Trading`, `RL Training & Callbacks`?**
  _High betweenness centrality (0.127) - this node is a cross-community bridge._
- **Why does `main()` connect `Live Paper Trading` to `RL Training & Callbacks`?**
  _High betweenness centrality (0.107) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `SMCTradingEnv` (e.g. with `live_trader.py — Paper Trading Engine for SMC PPO Model  Runs the trained RL age` and `Fetch the last `limit` candles from Binance (no API key needed for public).`) actually correct?**
  _`SMCTradingEnv` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `main()` (e.g. with `analyze_chart()` and `SMCTradingEnv`) actually correct?**
  _`main()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `run_one_generation()` (e.g. with `diagnose()` and `tournament()`) actually correct?**
  _`run_one_generation()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `main()` (e.g. with `get_data()` and `analyze_chart()`) actually correct?**
  _`main()` has 3 INFERRED edges - model-reasoned connections that need verification._