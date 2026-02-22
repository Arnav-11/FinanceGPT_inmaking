"""
portfolio.py
------------
Virtual portfolio simulator.
Stores portfolio in data_storage/portfolio.json
Supports buy/sell, tracks P&L, calculates returns.
"""

import os
import json
import logging
from datetime import datetime

log = logging.getLogger(__name__)

_BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORTFOLIO_FILE = os.path.join(_BASE_DIR, "data_storage", "portfolio.json")
INITIAL_CASH   = 100_000.0   # ₹1,00,000 virtual capital


def _load() -> dict:
    if not os.path.exists(PORTFOLIO_FILE):
        return {
            "cash":        INITIAL_CASH,
            "holdings":    {},   # symbol -> {shares, avg_price, invested}
            "transactions": [],  # list of trade history
            "created_at":  datetime.now().isoformat(),
        }
    with open(PORTFOLIO_FILE, "r") as f:
        return json.load(f)


def _save(data: dict):
    os.makedirs(os.path.dirname(PORTFOLIO_FILE), exist_ok=True)
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_portfolio() -> dict:
    """Return current portfolio with live P&L."""
    portfolio = _load()

    # Try to get live prices for each holding
    total_invested = 0.0
    total_current  = 0.0
    holdings_detail = []

    for symbol, h in portfolio["holdings"].items():
        invested = h["avg_price"] * h["shares"]
        total_invested += invested

        # Try live price
        current_price = h["avg_price"]  # fallback to avg
        try:
            from data.stock_data import get_live_price
            result = get_live_price(symbol)
            if "price" in result:
                current_price = result["price"]
        except Exception:
            pass

        current_value = current_price * h["shares"]
        total_current += current_value
        pnl           = current_value - invested
        pnl_pct       = round((pnl / invested) * 100, 2) if invested else 0

        holdings_detail.append({
            "symbol":        symbol,
            "name":          h.get("name", symbol),
            "shares":        h["shares"],
            "avg_price":     round(h["avg_price"], 2),
            "current_price": round(current_price, 2),
            "invested":      round(invested, 2),
            "current_value": round(current_value, 2),
            "pnl":           round(pnl, 2),
            "pnl_pct":       pnl_pct,
        })

    # Sort by P&L descending
    holdings_detail.sort(key=lambda x: x["pnl"], reverse=True)

    total_pnl     = total_current - total_invested
    total_pnl_pct = round((total_pnl / total_invested) * 100, 2) if total_invested else 0
    total_value   = portfolio["cash"] + total_current
    total_return  = round(((total_value - INITIAL_CASH) / INITIAL_CASH) * 100, 2)

    return {
        "cash":            round(portfolio["cash"], 2),
        "initial_capital": INITIAL_CASH,
        "total_value":     round(total_value, 2),
        "total_invested":  round(total_invested, 2),
        "total_current":   round(total_current, 2),
        "total_pnl":       round(total_pnl, 2),
        "total_pnl_pct":   total_pnl_pct,
        "total_return_pct": total_return,
        "holdings":        holdings_detail,
        "transaction_count": len(portfolio["transactions"]),
    }


def buy_stock(symbol: str, shares: int, name: str = "") -> dict:
    """Buy shares of a stock at current live price."""
    if shares <= 0:
        return {"error": "Shares must be a positive integer"}

    # Get current price
    try:
        from data.stock_data import get_live_price
        result = get_live_price(symbol)
        if "error" in result:
            return {"error": f"Could not get price for {symbol}: {result['error']}"}
        price = result["price"]
    except Exception as e:
        return {"error": str(e)}

    portfolio  = _load()
    total_cost = price * shares

    if total_cost > portfolio["cash"]:
        return {
            "error": f"Insufficient funds. Need ₹{total_cost:,.2f}, have ₹{portfolio['cash']:,.2f}"
        }

    # Update holdings
    if symbol in portfolio["holdings"]:
        h         = portfolio["holdings"][symbol]
        total_shares = h["shares"] + shares
        total_inv    = (h["avg_price"] * h["shares"]) + total_cost
        h["shares"]    = total_shares
        h["avg_price"]  = total_inv / total_shares
        if name:
            h["name"] = name
    else:
        portfolio["holdings"][symbol] = {
            "shares":    shares,
            "avg_price": price,
            "name":      name or symbol,
        }

    portfolio["cash"] -= total_cost
    portfolio["transactions"].append({
        "type":      "BUY",
        "symbol":    symbol,
        "shares":    shares,
        "price":     price,
        "total":     round(total_cost, 2),
        "timestamp": datetime.now().isoformat(),
    })

    _save(portfolio)

    return {
        "message":    f"Bought {shares} shares of {symbol} at ₹{price}",
        "shares":     shares,
        "price":      price,
        "total_cost": round(total_cost, 2),
        "cash_left":  round(portfolio["cash"], 2),
    }


def sell_stock(symbol: str, shares: int) -> dict:
    """Sell shares of a stock at current live price."""
    if shares <= 0:
        return {"error": "Shares must be a positive integer"}

    portfolio = _load()

    if symbol not in portfolio["holdings"]:
        return {"error": f"{symbol} is not in your portfolio"}

    h = portfolio["holdings"][symbol]
    if shares > h["shares"]:
        return {"error": f"You only have {h['shares']} shares of {symbol}"}

    # Get current price
    try:
        from data.stock_data import get_live_price
        result = get_live_price(symbol)
        if "error" in result:
            return {"error": f"Could not get price for {symbol}: {result['error']}"}
        price = result["price"]
    except Exception as e:
        return {"error": str(e)}

    proceeds  = price * shares
    pnl       = (price - h["avg_price"]) * shares

    h["shares"] -= shares
    if h["shares"] == 0:
        del portfolio["holdings"][symbol]

    portfolio["cash"] += proceeds
    portfolio["transactions"].append({
        "type":      "SELL",
        "symbol":    symbol,
        "shares":    shares,
        "price":     price,
        "total":     round(proceeds, 2),
        "pnl":       round(pnl, 2),
        "timestamp": datetime.now().isoformat(),
    })

    _save(portfolio)

    return {
        "message":  f"Sold {shares} shares of {symbol} at ₹{price}",
        "shares":   shares,
        "price":    price,
        "proceeds": round(proceeds, 2),
        "pnl":      round(pnl, 2),
        "cash":     round(portfolio["cash"], 2),
    }


def reset_portfolio() -> dict:
    """Reset portfolio back to initial ₹1,00,000 cash."""
    data = {
        "cash":         INITIAL_CASH,
        "holdings":     {},
        "transactions": [],
        "created_at":   datetime.now().isoformat(),
    }
    _save(data)
    return {"message": f"Portfolio reset to ₹{INITIAL_CASH:,.0f}"}


def get_transactions() -> list:
    """Return full transaction history."""
    return _load().get("transactions", [])