def analyze_signal(df):
    if len(df) < 25:
        return {"valid": False, "direction": "N/A", "score": 0}

    last = df.iloc[-1]
    prev1 = df.iloc[-2]
    prev2 = df.iloc[-3]

    # 1. Tendance
    trend_up = last['ema_20'] > last['ema_50']
    ema_20_slope_up = df['ema_20'].iloc[-1] > df['ema_20'].iloc[-2] > df['ema_20'].iloc[-3]
    ema_trend_ok = trend_up and ema_20_slope_up

    # 2. Momentum
    rsi_long = 50 < last['rsi'] < 70
    rsi_short = 30 < last['rsi'] < 50
    macd_ok = (last['macd'] > last['signal']) and (last['macd'] - last['signal'] > prev1['macd'] - prev1['signal'])

    # 3. Volatilité
    vol_ok = last['volatility'] > df['volatility'].rolling(20).mean().iloc[-1]
    return_pos = last['returns'] > 0 if trend_up else last['returns'] < 0

    # Score de confluence
    score = sum([ema_trend_ok, (rsi_long if trend_up else rsi_short), macd_ok, vol_ok and return_pos])

    return {
        "valid": score == 4,
        "direction": "LONG" if trend_up else "SHORT",
        "score": score,
        "rsi": last['rsi'],
        "price": last['close']
    }
