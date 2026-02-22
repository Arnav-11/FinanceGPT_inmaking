"""
screener.py  (place in: services/screener.py)
-----------------------------------------------
Stock Screener - scan all 151 Indian stocks and filter by
technical indicator conditions.

API usage examples:
  GET /screener?rsi_below=35&macd_bullish=true
  GET /screener?verdict=STRONG+BUY&sector=IT
  GET /screener?near_52w_low=5&volume_spike=2.0
  GET /screener/presets/oversold-bullish
"""

import logging
from typing import Optional

log = logging.getLogger(__name__)


def run_screener(
    rsi_below:     Optional[float] = None,   # RSI < value
    rsi_above:     Optional[float] = None,   # RSI > value
    macd_bullish:  Optional[bool]  = None,   # MACD > Signal
    macd_bearish:  Optional[bool]  = None,   # MACD < Signal
    above_ma50:    Optional[bool]  = None,   # Price > MA50
    below_ma50:    Optional[bool]  = None,   # Price < MA50
    above_ma20:    Optional[bool]  = None,   # Price > MA20
    below_ma20:    Optional[bool]  = None,   # Price < MA20
    near_52w_high: Optional[float] = None,   # within N% of 52w high
    near_52w_low:  Optional[float] = None,   # within N% of 52w low
    volume_spike:  Optional[float] = None,   # > N x avg volume
    bb_oversold:   Optional[bool]  = None,   # price < lower BB
    bb_overbought: Optional[bool]  = None,   # price > upper BB
    verdict:       Optional[str]   = None,   # STRONG BUY / BUY / HOLD / SELL / STRONG SELL
    sector:        Optional[str]   = None,   # filter by sector
    min_score:     Optional[int]   = None,   # minimum analyst score
    max_score:     Optional[int]   = None,   # maximum analyst score
    limit:         int             = 50,
) -> dict:
    """
    Scan all stocks and return those matching all specified filters.
    Results are sorted by analyst score (best first).
    """
    try:
        from stocks_list import INDIAN_STOCKS
        from data.stock_data import get_historical_data
        from indicators.technicals import calculate_rsi, calculate_macd, moving_average, bollinger_bands
        from services.analysis import generate_summary
    except Exception as e:
        return {"error": f"Import error: {str(e)}"}

    stocks = [s for s in INDIAN_STOCKS if s["index"] != "INDEX"]
    if sector:
        stocks = [s for s in stocks if s["sector"].lower() == sector.lower()]

    matched = []
    scanned = 0
    skipped = 0

    for stock in stocks:
        symbol = stock["symbol"]
        df = get_historical_data(symbol)
        if df is None or len(df) < 60:
            skipped += 1
            continue

        scanned += 1

        try:
            rsi   = calculate_rsi(df)
            macd  = calculate_macd(df)
            ma20  = moving_average(df, 20)
            ma50  = moving_average(df, 50)
            bb    = bollinger_bands(df)
            price = float(df["Close"].iloc[-1])

            year_close = df["Close"].tail(252)
            high_52w   = float(year_close.max())
            low_52w    = float(year_close.min())

            vol_today  = float(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0
            vol_avg    = float(df["Volume"].tail(20).mean()) if "Volume" in df.columns else 1
            vol_ratio  = round(vol_today / vol_avg, 2) if vol_avg > 0 else 0

            summary = generate_summary(rsi, macd, price, ma20, ma50, bb)
            score   = summary["score"]
            sv      = summary["verdict"]

            # Apply all filters — skip if any condition fails
            if rsi_below     is not None and rsi >= rsi_below:             continue
            if rsi_above     is not None and rsi <= rsi_above:             continue
            if macd_bullish  and macd["macd"] <= macd["signal"]:           continue
            if macd_bearish  and macd["macd"] >= macd["signal"]:           continue
            if above_ma50    and price <= ma50:                            continue
            if below_ma50    and price >= ma50:                            continue
            if above_ma20    and price <= ma20:                            continue
            if below_ma20    and price >= ma20:                            continue
            if bb_oversold   and price >= bb["lower"]:                     continue
            if bb_overbought and price <= bb["upper"]:                     continue
            if verdict       and sv.upper() != verdict.upper():            continue
            if min_score     is not None and score < min_score:            continue
            if max_score     is not None and score > max_score:            continue
            if volume_spike  is not None and vol_ratio < volume_spike:     continue

            if near_52w_high is not None:
                if abs((price - high_52w) / high_52w * 100) > near_52w_high:  continue
            if near_52w_low is not None:
                if abs((price - low_52w) / low_52w * 100) > near_52w_low:     continue

            pfh = round((price - high_52w) / high_52w * 100, 2)
            pfl = round((price - low_52w)  / low_52w  * 100, 2)

            matched.append({
                "symbol":        symbol,
                "name":          stock["name"],
                "sector":        stock["sector"],
                "index":         stock["index"],
                "price":         round(price, 2),
                "rsi":           rsi,
                "macd":          macd["macd"],
                "macd_signal":   macd["signal"],
                "macd_bullish":  macd["macd"] > macd["signal"],
                "ma20":          ma20,
                "ma50":          ma50,
                "above_ma20":    price > ma20,
                "above_ma50":    price > ma50,
                "bb_upper":      bb["upper"],
                "bb_lower":      bb["lower"],
                "high_52w":      round(high_52w, 2),
                "low_52w":       round(low_52w,  2),
                "pct_from_high": f"{pfh}%",
                "pct_from_low":  f"+{pfl}%" if pfl >= 0 else f"{pfl}%",
                "volume_ratio":  vol_ratio,
                "verdict":       sv,
                "score":         score,
                "signals":       summary["signals"],
            })

        except Exception as e:
            log.debug(f"Skipped {symbol}: {e}")
            skipped += 1

    matched.sort(key=lambda x: x["score"], reverse=True)

    filters_used = {}
    for k, v in [
        ("rsi_below", rsi_below), ("rsi_above", rsi_above),
        ("macd_bullish", macd_bullish), ("macd_bearish", macd_bearish),
        ("above_ma50", above_ma50), ("below_ma50", below_ma50),
        ("above_ma20", above_ma20), ("below_ma20", below_ma20),
        ("near_52w_high", near_52w_high), ("near_52w_low", near_52w_low),
        ("volume_spike", volume_spike), ("bb_oversold", bb_oversold),
        ("bb_overbought", bb_overbought), ("verdict", verdict),
        ("sector", sector), ("min_score", min_score), ("max_score", max_score),
    ]:
        if v is not None:
            filters_used[k] = v

    return {
        "total_scanned":   scanned,
        "total_matched":   len(matched),
        "total_skipped":   skipped,
        "filters_applied": filters_used,
        "results":         matched[:limit],
    }


# Convenience preset screeners
def screen_oversold_bullish(limit=20):
    """RSI < 35 + MACD bullish crossover."""
    return run_screener(rsi_below=35, macd_bullish=True, limit=limit)

def screen_strong_uptrend(limit=20):
    """Price above MA20 + MA50 + MACD bullish."""
    return run_screener(above_ma20=True, above_ma50=True, macd_bullish=True, limit=limit)

def screen_breakout_candidates(limit=20):
    """Within 3% of 52w high + volume spike > 1.5x."""
    return run_screener(near_52w_high=3.0, volume_spike=1.5, limit=limit)

def screen_value_picks(limit=20):
    """Within 10% of 52w low + RSI not overbought."""
    return run_screener(near_52w_low=10.0, rsi_below=50, limit=limit)

def screen_strong_buys(limit=20):
    """All STRONG BUY rated stocks."""
    return run_screener(verdict="STRONG BUY", limit=limit)