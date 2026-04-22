import pandas as pd
import numpy as np
import pandas_ta as ta

PARAMS = {
    "atr_period": 14,      # Period for ATR calculation.
    "fvg_atr_multi": 0.55, # ADJUSTED: Decreased FVG minimum size to allow more trades and improve generalization.
    "sl_atr_multi": 2.0,   # Retained: Standard risk management.
    "tp_atr_multi": 5.0,   # Retained: Targets a theoretical 2.5 R:R.
    "trail_atr_multi": 2.0, # Retained: For trailing stop logic outside this function.
    "sma_fast_period": 20, # Retained: Period for the faster Simple Moving Average.
    "sma_slow_period": 50, # ADJUSTED: Increased slow SMA period for a more robust trend filter.
    "cmf_period": 20,      # Retained: Period for Chaikin Money Flow.
    "cmf_threshold": 0.05  # NEW PARAMETER: Loosened CMF threshold for more inclusive volume confirmation.
}

def analyze_chart(df):
    """
    Inputs OHLCV DataFrame and appends Smart Money Concepts signals purely natively.
    Refined with a simplified retest-based FVG entry strategy for better generalization,
    aiming for improved profit factor and reduced overfitting by reducing condition complexity.
    """
    # Ensure required columns are present
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain {required_columns}")

    # 1. Calculate Simple Moving Averages for trend filtering
    df['SMA_FAST'] = ta.sma(df['Close'], length=PARAMS['sma_fast_period'])
    df['SMA_SLOW'] = ta.sma(df['Close'], length=PARAMS['sma_slow_period'])

    # 2. Chaikin Money Flow (CMF) for volume-price confirmation
    df['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=PARAMS['cmf_period'])

    # 3. Average True Range (ATR) for dynamic sizing
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=PARAMS['atr_period'])

    # 4. Fair Value Gap (FVG) Detection - identifying the *imbalance itself*
    # Bullish FVG occurs when Low[t] > High[t-2]. The gap is between High[t-2] and Low[t].
    df['bull_fvg_size_raw'] = df['Low'] - df['High'].shift(2)
    # Bearish FVG occurs when High[t] < Low[t-2]. The gap is between Low[t-2] and High[t].
    df['bear_fvg_size_raw'] = df['Low'].shift(2) - df['High']

    # Determine if an FVG is 'active' based on minimum size requirement (now dynamic using ATR)
    df['bull_fvg_active'] = (df['Low'] > df['High'].shift(2)) & \
                            (df['bull_fvg_size_raw'] >= PARAMS['fvg_atr_multi'] * df['ATR'])
    df['bear_fvg_active'] = (df['High'] < df['Low'].shift(2)) & \
                            (df['bear_fvg_size_raw'] >= PARAMS['fvg_atr_multi'] * df['ATR'])

    # Capture the FVG bounds *when they are formed* for subsequent retest checks
    # For a bullish FVG formed at bar 't', the void is from High[t-2] (upper level) to Low[t] (lower level).
    df['fvg_bull_upper_level_at_creation'] = df['High'].shift(2) # Upper boundary of the bullish FVG void
    df['fvg_bull_lower_level_at_creation'] = df['Low']          # Lower boundary of the bullish FVG void

    # For a bearish FVG formed at bar 't', the void is from High[t] (upper level) to Low[t-2] (lower level).
    df['fvg_bear_upper_level_at_creation'] = df['High']          # Upper boundary of the bearish FVG void
    df['fvg_bear_lower_level_at_creation'] = df['Low'].shift(2) # Lower boundary of the bearish FVG void

    # Initialize signal column
    df['signal'] = 0

    # 5. Establish Positions with FVG Retest + Trend + CMF Confirmation
    # Conditions are now precisely targeting 'retest and confirmed bounce/rejection' (5 conditions).

    # Long signal: Retest of a previous bullish FVG
    # A bullish FVG was formed at the previous bar (t-1). The current bar (t) retraces
    # to its lower boundary and bounces, closing *above* the FVG boundary.
    long_cond = (
        df['bull_fvg_active'].shift(1).fillna(False) &                           # 1. A bullish FVG was active at t-1
        (df['Low'] <= df['fvg_bull_lower_level_at_creation'].shift(1)).fillna(False) & # 2. Current low touches/enters the lower FVG boundary
        (df['Close'] > df['fvg_bull_lower_level_at_creation'].shift(1)).fillna(False) & # 3. Current candle closes *above* FVG's lower boundary (confirms bounce)
        (df['SMA_FAST'] > df['SMA_SLOW']).fillna(False) &                         # 4. Strong uptrend context
        (df['CMF'] > PARAMS['cmf_threshold']).fillna(False)                        # 5. Volume confirmation, using new CMF threshold
    )
    df.loc[long_cond, 'signal'] = 1

    # Short signal: Retest of a previous bearish FVG
    # A bearish FVG was formed at the previous bar (t-1). The current bar (t) retraces
    # to its upper boundary and rejects, closing *below* the FVG boundary.
    short_cond = (
        df['bear_fvg_active'].shift(1).fillna(False) &                           # 1. A bearish FVG was active at t-1
        (df['High'] >= df['fvg_bear_upper_level_at_creation'].shift(1)).fillna(False) & # 2. Current high touches/enters the upper FVG boundary
        (df['Close'] < df['fvg_bear_upper_level_at_creation'].shift(1)).fillna(False) & # 3. Current candle closes *below* FVG's upper boundary (confirms rejection)
        (df['SMA_FAST'] < df['SMA_SLOW']).fillna(False) &                         # 4. Strong downtrend context
        (df['CMF'] < -PARAMS['cmf_threshold']).fillna(False)                       # 5. Volume confirmation, using new CMF threshold
    )
    df.loc[short_cond, 'signal'] = -1
    
    # Clean up auxiliary columns - COMMENTED OUT FOR RL TRAINING FEATURE VISIBILITY
    # df.drop(columns=[
    #     'SMA_FAST', 'SMA_SLOW', 'CMF', 'ATR',
    #     'bull_fvg_size_raw', 'bear_fvg_size_raw',
    #     'bull_fvg_active', 'bear_fvg_active',
    #     'fvg_bull_upper_level_at_creation', 'fvg_bull_lower_level_at_creation',
    #     'fvg_bear_upper_level_at_creation', 'fvg_bear_lower_level_at_creation'
    # ], inplace=True, errors='ignore')
    
    return df
