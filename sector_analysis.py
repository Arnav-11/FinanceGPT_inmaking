"""
sector_analysis.py
------------------
Sector-level analysis — aggregates stock signals
to produce a heatmap-ready sector summary.
"""

import os
import logging
from collections import defaultdict

log = logging.getLogger(__name__)


def get_sector_heatmap() -> list:
    """
    Analyze all stocks grouped by sector.
    Returns list of sector summaries sorted by average score.
    """
    try:
        from stocks_list import INDIAN_STOCKS
        from services.analysis import analyze_stock, generate_summary
        from data.stock_data import get_historical_data
        from indicators.technicals import calculate_rsi, calculate_macd, moving_average, bollinger_bands
    except Exception as e:
        return [{"error": str(e)}]

    sector_data = defaultdict(list)

    # Group stocks by sector
    stocks = [s for s in INDIAN_STOCKS if s["index"] != "INDEX"]

    for stock in stocks:
        symbol = stock["symbol"]
        sector = stock["sector"]

        df = get_historical_data(symbol)
        if df is None:
            continue

        try:
            rsi   = calculate_rsi(df)
            macd  = calculate_macd(df)
            ma20  = moving_average(df, 20)
            ma50  = moving_average(df, 50)
            bb    = bollinger_bands(df)
            price = float(df["Close"].iloc[-1])

            summary = generate_summary(rsi, macd, price, ma20, ma50, bb)
            sector_data[sector].append({
                "symbol":  symbol,
                "name":    stock["name"],
                "score":   summary["score"],
                "verdict": summary["verdict"],
                "price":   round(price, 2),
            })
        except Exception:
            continue

    # Aggregate by sector
    result = []
    for sector, stocks_in_sector in sector_data.items():
        if not stocks_in_sector:
            continue

        scores   = [s["score"] for s in stocks_in_sector]
        avg_score = round(sum(scores) / len(scores), 2)

        # Count verdicts
        verdicts = defaultdict(int)
        for s in stocks_in_sector:
            verdicts[s["verdict"]] += 1

        # Overall sector verdict
        if avg_score >= 3:
            sector_verdict = "BULLISH"
        elif avg_score >= 1:
            sector_verdict = "MILDLY BULLISH"
        elif avg_score <= -3:
            sector_verdict = "BEARISH"
        elif avg_score <= -1:
            sector_verdict = "MILDLY BEARISH"
        else:
            sector_verdict = "NEUTRAL"

        result.append({
            "sector":        sector,
            "avg_score":     avg_score,
            "verdict":       sector_verdict,
            "stock_count":   len(stocks_in_sector),
            "verdicts":      dict(verdicts),
            "top_stocks":    sorted(stocks_in_sector,
                                   key=lambda x: x["score"],
                                   reverse=True)[:3],
        })

    # Sort by avg_score descending (best sectors first)
    result.sort(key=lambda x: x["avg_score"], reverse=True)
    return result