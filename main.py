import math
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from data.stock_data import get_live_price, get_historical_data
from ml_models.dataset_loader import download_data
from services.analysis import analyze_stock
from services.watchlist import get_watchlist, add_to_watchlist, remove_from_watchlist
from ml_models.predict import predict
from ml_models.backtest import backtest
from stocks_list import INDIAN_STOCKS

log = logging.getLogger(__name__)

app = FastAPI(title="FinanceGPT API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/ui")
def ui():
    return FileResponse("frontend/index.html")


@app.get("/")
def root():
    return {"status": "ok", "message": "FinanceGPT API v3.0 is running"}


# ── Stocks list ───────────────────────────────────────────────────────────────

@app.get("/stocks/list")
def stocks_list(sector: str = None, index: str = None):
    stocks = INDIAN_STOCKS
    if sector:
        stocks = [s for s in stocks if s["sector"].lower() == sector.lower()]
    if index:
        stocks = [s for s in stocks if s["index"].lower() == index.lower()]
    return stocks


# ── Data endpoints ────────────────────────────────────────────────────────────

@app.post("/data/download/{symbol}")
def api_download(symbol: str, period: str = "5y"):
    try:
        path = download_data(symbol, period)
        return {"message": f"Data saved to {path}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/data/download-all")
def api_download_all(period: str = "5y", workers: int = 4):
    try:
        from ml_models.bulk_downloader import bulk_download
        results = bulk_download(period=period, workers=workers)
        return {
            "downloaded":     len(results["success"]),
            "cached":         len(results["cached"]),
            "failed":         len(results["failed"]),
            "failed_symbols": [s for s, _ in results["failed"]],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/price/{symbol}")
def api_price(symbol: str):
    result = get_live_price(symbol)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/data/chart/{symbol}")
def api_chart(symbol: str, days: int = 180):
    """OHLCV + MA overlay for charting — includes candlestick data."""
    df = get_historical_data(symbol)
    if df is None:
        raise HTTPException(
            status_code=404,
            detail=f"No cached data for '{symbol}'. Call POST /data/download/{symbol} first."
        )

    df = df.tail(days).copy()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()

    records = []
    for date, row in df.iterrows():
        ma20   = None if math.isnan(row["MA_20"]) else round(float(row["MA_20"]), 2)
        ma50   = None if math.isnan(row["MA_50"]) else round(float(row["MA_50"]), 2)
        volume = int(row["Volume"]) if "Volume" in row and not math.isnan(row["Volume"]) else None
        records.append({
            "date":   str(date)[:10],
            "open":   round(float(row["Open"]),  2) if "Open"  in row else None,
            "high":   round(float(row["High"]),  2) if "High"  in row else None,
            "low":    round(float(row["Low"]),   2) if "Low"   in row else None,
            "close":  round(float(row["Close"]), 2),
            "volume": volume,
            "ma20":   ma20,
            "ma50":   ma50,
        })
    return records


# ── Analysis ──────────────────────────────────────────────────────────────────

@app.get("/analysis/{symbol}")
def api_analysis(symbol: str):
    result = analyze_stock(symbol)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/compare")
def api_compare(symbol1: str, symbol2: str):
    r1 = analyze_stock(symbol1)
    r2 = analyze_stock(symbol2)
    return {"stock1": r1, "stock2": r2}


# ── ML endpoints ──────────────────────────────────────────────────────────────

@app.post("/ml/train/general")
def api_train_general(download_fresh: bool = False):
    try:
        from ml_models.train_model import train_general
        accuracy = train_general(download_missing=download_fresh)
        return {
            "message":  "Generalized market model trained successfully",
            "accuracy": round(accuracy, 4),
            "model":    "stock_model_general.pkl",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ml/train/{symbol}")
def api_train(symbol: str, download_fresh: bool = False):
    try:
        from ml_models.train_model import train
        accuracy = train(symbol=symbol, download=download_fresh)
        return {
            "message":  f"Per-symbol model trained for {symbol}",
            "accuracy": round(accuracy, 4),
            "model":    f"models/stock_model_{symbol.replace('.','_')}.pkl",
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ml/predict/{symbol}")
def api_predict(symbol: str):
    try:
        return predict(symbol)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/backtest/{symbol}")
def api_backtest(symbol: str, capital: float = 100000.0):
    try:
        result = backtest(symbol, initial_capital=capital)
        return result["summary"]
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── LSTM endpoints ────────────────────────────────────────────────────────────

@app.post("/ml/train-lstm/general")
def api_train_lstm_general(download_fresh: bool = False):
    try:
        from ml_models.lstm_model import train_general as lstm_train_general
        accuracy = lstm_train_general(download_missing=download_fresh)
        return {
            "message":  "Generalized LSTM model trained successfully",
            "accuracy": round(accuracy, 4),
            "model":    "lstm_model_general.keras",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ml/train-lstm/{symbol}")
def api_train_lstm_symbol(symbol: str, download_fresh: bool = False):
    try:
        from ml_models.lstm_model import train as lstm_train
        accuracy = lstm_train(symbol=symbol, download=download_fresh)
        return {
            "message":  f"Per-symbol LSTM model trained for {symbol}",
            "accuracy": round(accuracy, 4),
            "model":    f"models/lstm_model_{symbol.replace('.','_')}.keras",
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ml/predict-lstm/{symbol}")
def api_predict_lstm(symbol: str):
    try:
        from ml_models.lstm_model import predict_lstm
        return predict_lstm(symbol)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ml/predict-combined/{symbol}")
def api_predict_combined(symbol: str):
    """Combined ML + LSTM ensemble — the most powerful prediction endpoint."""
    try:
        from ml_models.lstm_model import predict_combined
        return predict_combined(symbol)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Phase 3: Sentiment ────────────────────────────────────────────────────────

@app.get("/sentiment/{symbol}")
def api_sentiment(symbol: str):
    """Fetch and score latest news headlines for a stock."""
    try:
        from services.sentiment import get_sentiment
        return get_sentiment(symbol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Phase 3: Full analysis with sentiment + ML ────────────────────────────────

@app.get("/full-analysis/{symbol}")
def api_full_analysis(symbol: str):
    """
    Complete analysis combining:
    - Technical indicators
    - ML prediction
    - LSTM prediction (if available)
    - News sentiment
    - Explainability (top features)
    """
    result = {}

    # Technical analysis
    try:
        result["technical"] = analyze_stock(symbol)
    except Exception as e:
        result["technical"] = {"error": str(e)}

    # ML prediction
    try:
        result["ml_prediction"] = predict(symbol)
    except Exception as e:
        result["ml_prediction"] = {"error": str(e)}

    # LSTM + combined prediction
    try:
        from ml_models.lstm_model import predict_combined
        result["combined_prediction"] = predict_combined(symbol)
    except Exception as e:
        result["combined_prediction"] = {"error": str(e)}

    # News sentiment
    try:
        from services.sentiment import get_sentiment
        result["sentiment"] = get_sentiment(symbol)
    except Exception as e:
        result["sentiment"] = {"error": str(e)}

    # Final verdict combining all signals
    try:
        tech_verdict = result.get("technical", {}).get("recommendation", "HOLD")
        ml_direction = result.get("ml_prediction", {}).get("prediction", "")
        sentiment_label = result.get("sentiment", {}).get("overall_label", "NEUTRAL")

        signals_agree = (
            (tech_verdict in ["BUY", "STRONG BUY"] and ml_direction == "UP") or
            (tech_verdict in ["SELL", "STRONG SELL"] and ml_direction == "DOWN")
        )

        result["meta"] = {
            "symbol":           symbol,
            "technical_verdict": tech_verdict,
            "ml_direction":      ml_direction,
            "sentiment":         sentiment_label,
            "signals_agree":     signals_agree,
            "conviction":        "HIGH" if signals_agree else "LOW",
        }
    except Exception:
        pass

    return result


# ── Phase 3: Sector heatmap ───────────────────────────────────────────────────

@app.get("/sectors/heatmap")
def api_sector_heatmap():
    """Sector-level sentiment and signal heatmap."""
    try:
        from services.sector_analysis import get_sector_heatmap
        return get_sector_heatmap()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Phase 3: Portfolio ────────────────────────────────────────────────────────

@app.get("/portfolio")
def api_get_portfolio():
    """Get current virtual portfolio with live P&L."""
    try:
        from services.portfolio import get_portfolio
        return get_portfolio()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/portfolio/buy/{symbol}")
def api_buy(symbol: str, shares: int, name: str = ""):
    try:
        from services.portfolio import buy_stock
        result = buy_stock(symbol, shares, name)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/portfolio/sell/{symbol}")
def api_sell(symbol: str, shares: int):
    try:
        from services.portfolio import sell_stock
        result = sell_stock(symbol, shares)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/portfolio/reset")
def api_reset_portfolio():
    try:
        from services.portfolio import reset_portfolio
        return reset_portfolio()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/portfolio/transactions")
def api_transactions():
    try:
        from services.portfolio import get_transactions
        return get_transactions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Phase 3: Risk Profile ─────────────────────────────────────────────────────

class RiskProfileInput(BaseModel):
    horizon:   str    # "short" | "medium" | "long"
    tolerance: str    # "low" | "medium" | "high"
    capital:   float  # amount in INR


@app.post("/risk-profile")
def api_set_risk_profile(body: RiskProfileInput):
    """Set user risk profile based on 3 questions."""
    try:
        from services.risk_profile import set_risk_profile
        result = set_risk_profile(body.horizon, body.tolerance, body.capital)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/risk-profile")
def api_get_risk_profile():
    """Get current saved risk profile."""
    try:
        from services.risk_profile import get_risk_profile
        return get_risk_profile()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/risk-profile/recommendations")
def api_risk_recommendations():
    """
    Get stock recommendations filtered by your risk profile.
    Analyzes all watchlist stocks and returns only those matching your profile.
    """
    try:
        from services.risk_profile import get_risk_profile, filter_by_risk
        from services.watchlist import get_watchlist

        profile = get_risk_profile()
        if "error" in profile:
            raise HTTPException(status_code=400, detail=profile["error"])

        watchlist = get_watchlist()
        if not watchlist:
            return {"message": "Your watchlist is empty. Add stocks first.", "recommendations": []}

        recommendations = []
        for item in watchlist:
            symbol = item["symbol"]
            analysis = analyze_stock(symbol)
            if "error" not in analysis:
                recommendations.append({
                    "symbol":         symbol,
                    "name":           item.get("name", symbol),
                    "recommendation": analysis.get("recommendation", "HOLD"),
                    "price":          analysis.get("price"),
                    "trend":          analysis.get("trend"),
                    "score":          analysis.get("analyst_summary", {}).get("score"),
                })

        filtered = filter_by_risk(recommendations)
        return {
            "profile":         profile["label"],
            "total_stocks":    len(recommendations),
            "matching_stocks": len(filtered),
            "recommendations": filtered,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Watchlist ─────────────────────────────────────────────────────────────────

@app.get("/watchlist")
def api_get_watchlist():
    return get_watchlist()


@app.post("/watchlist/{symbol}")
def api_add_watchlist(symbol: str, name: str = ""):
    return add_to_watchlist(symbol, name)


@app.delete("/watchlist/{symbol}")
def api_remove_watchlist(symbol: str):
    return remove_from_watchlist(symbol)


# ============================================================
# ADD THESE ROUTES TO YOUR EXISTING main.py
# Place them at the bottom, before the last few lines
# ============================================================

# ── Screener ──────────────────────────────────────────────────────────────────

@app.get("/screener")
def api_screener(
    rsi_below:     float = None,
    rsi_above:     float = None,
    macd_bullish:  bool  = None,
    macd_bearish:  bool  = None,
    above_ma50:    bool  = None,
    below_ma50:    bool  = None,
    above_ma20:    bool  = None,
    below_ma20:    bool  = None,
    near_52w_high: float = None,
    near_52w_low:  float = None,
    volume_spike:  float = None,
    bb_oversold:   bool  = None,
    bb_overbought: bool  = None,
    verdict:       str   = None,
    sector:        str   = None,
    min_score:     int   = None,
    max_score:     int   = None,
    limit:         int   = 50,
):
    """
    Scan all stocks with optional technical filters.

    Examples:
      /screener?rsi_below=35&macd_bullish=true
      /screener?verdict=STRONG+BUY&sector=IT
      /screener?near_52w_low=5&volume_spike=2.0
      /screener?above_ma50=true&above_ma20=true&macd_bullish=true
    """
    try:
        from services.screener import run_screener
        return run_screener(
            rsi_below=rsi_below, rsi_above=rsi_above,
            macd_bullish=macd_bullish, macd_bearish=macd_bearish,
            above_ma50=above_ma50, below_ma50=below_ma50,
            above_ma20=above_ma20, below_ma20=below_ma20,
            near_52w_high=near_52w_high, near_52w_low=near_52w_low,
            volume_spike=volume_spike, bb_oversold=bb_oversold,
            bb_overbought=bb_overbought, verdict=verdict,
            sector=sector, min_score=min_score, max_score=max_score,
            limit=limit,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/screener/presets/{preset}")
def api_screener_preset(preset: str):
    """
    Run a preset screener strategy.

    Available presets:
      oversold-bullish     - RSI < 35 + MACD bullish crossover
      strong-uptrend       - Price > MA20 + MA50 + MACD bullish
      breakout-candidates  - Within 3% of 52w high + volume spike
      value-picks          - Within 10% of 52w low + RSI < 50
      strong-buys          - All STRONG BUY rated stocks
    """
    try:
        from services.screener import (
            screen_oversold_bullish, screen_strong_uptrend,
            screen_breakout_candidates, screen_value_picks, screen_strong_buys
        )
        presets = {
            "oversold-bullish":    screen_oversold_bullish,
            "strong-uptrend":      screen_strong_uptrend,
            "breakout-candidates": screen_breakout_candidates,
            "value-picks":         screen_value_picks,
            "strong-buys":         screen_strong_buys,
        }
        if preset not in presets:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown preset '{preset}'. Available: {list(presets.keys())}"
            )
        return presets[preset]()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Portfolio Optimizer ───────────────────────────────────────────────────────

@app.get("/portfolio/optimize")
def api_optimize_portfolio(symbols: str = None):
    """
    Optimize portfolio allocation using Modern Portfolio Theory.

    Pass ?symbols=TCS.NS,INFY.NS,RELIANCE.NS to optimize a custom list.
    Leave empty to optimize your current virtual portfolio.

    Returns:
      - max_sharpe_portfolio  : best risk-adjusted return weights
      - min_volatility_portfolio : lowest risk weights
      - equal_weight_portfolio   : naive baseline
      - current_portfolio        : your current allocation stats
      - frontier_chart_data      : efficient frontier for visualization
      - individual_stock_stats   : per-stock return/risk/Sharpe
      - top_correlations         : most correlated pairs
      - low_correlations         : best diversification pairs
    """
    try:
        from services.portfolio_optimizer import optimize_portfolio
        symbol_list = [s.strip() for s in symbols.split(",")] if symbols else None
        result = optimize_portfolio(
            symbols=symbol_list,
            use_virtual_portfolio=(symbol_list is None),
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── AI Advisor (Chat) ─────────────────────────────────────────────────────────

@app.post("/advisor/chat")
def api_advisor_chat(question: str, session_id: str = "default"):
    """
    Ask the AI advisor a natural language finance question.

    Requires ANTHROPIC_API_KEY to be set in your environment.

    Example questions:
      "Is TCS a good buy right now?"
      "What should I do with my portfolio?"
      "Which IT stocks look bullish?"
      "Compare HDFC Bank vs ICICI Bank"
      "Optimize my portfolio allocation"

    session_id lets you maintain multi-turn conversations.
    Each session remembers the last 10 turns.
    """
    try:
        from services.advisor import chat
        result = chat(question=question, session_id=session_id)
        return result
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/advisor/session/{session_id}")
def api_clear_session(session_id: str):
    """Clear conversation history for a session."""
    try:
        from services.advisor import clear_session
        return clear_session(session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/advisor/history/{session_id}")
def api_get_history(session_id: str):
    """Get conversation history for a session."""
    try:
        from services.advisor import get_history
        return get_history(session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))