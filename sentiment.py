"""
sentiment.py
------------
News sentiment analysis for stocks.
Fetches headlines from Google News RSS (no API key needed)
and scores them using a financial keyword-based approach
+ TextBlob for general sentiment.

Falls back gracefully if network is unavailable.
"""

import re
import os
import json
import time
import logging
import urllib.request
import urllib.parse
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

# Financial positive / negative keyword lists
BULLISH_WORDS = {
    "surge", "surges", "surged", "rally", "rallies", "rallied",
    "gain", "gains", "gained", "rise", "rises", "rose", "risen",
    "jump", "jumps", "jumped", "soar", "soars", "soared",
    "profit", "profits", "profitable", "growth", "grew", "grow",
    "strong", "strength", "beat", "beats", "outperform",
    "upgrade", "upgraded", "buy", "record", "high", "boom",
    "positive", "bullish", "upside", "recovery", "recover",
    "dividend", "expansion", "revenue", "win", "wins", "won",
    "deal", "contract", "order", "orders", "acquisition",
}

BEARISH_WORDS = {
    "fall", "falls", "fell", "fallen", "drop", "drops", "dropped",
    "decline", "declines", "declined", "slide", "slides", "slid",
    "loss", "losses", "lose", "losing", "lost", "plunge", "plunges",
    "crash", "crashes", "crashed", "slump", "slumps", "slumped",
    "weak", "weakness", "miss", "misses", "missed", "disappoint",
    "downgrade", "downgraded", "sell", "low", "bear", "bearish",
    "debt", "default", "lawsuit", "fine", "penalty", "probe",
    "investigation", "fraud", "negative", "concern", "concerns",
    "risk", "risks", "warning", "warn", "cut", "cuts",
}


def _fetch_rss(symbol: str, company_name: str = "") -> list[dict]:
    """
    Fetch headlines from Google News RSS for a stock symbol.
    Returns list of {title, published} dicts.
    """
    # Build search query — use company name if available, else symbol
    query = company_name if company_name else symbol.replace(".NS", "").replace(".BO", "")
    query = f"{query} stock"
    encoded = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            content = resp.read().decode("utf-8")

        # Parse RSS items with regex (no external XML parser needed)
        titles     = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", content)
        pub_dates  = re.findall(r"<pubDate>(.*?)</pubDate>", content)

        # Skip first title (it's the feed title)
        titles = titles[1:16]  # max 15 headlines

        articles = []
        for i, title in enumerate(titles):
            pub = pub_dates[i] if i < len(pub_dates) else ""
            articles.append({"title": title.strip(), "published": pub})

        return articles

    except Exception as e:
        log.warning(f"Could not fetch news for {symbol}: {e}")
        return []


def _score_headline(title: str) -> float:
    """
    Score a single headline.
    Returns float in [-1, +1]:
      +1 = very bullish
      -1 = very bearish
       0 = neutral
    """
    words  = set(re.findall(r"\b\w+\b", title.lower()))
    bull   = len(words & BULLISH_WORDS)
    bear   = len(words & BEARISH_WORDS)
    total  = bull + bear
    if total == 0:
        return 0.0
    return round((bull - bear) / total, 3)


def _label(score: float) -> str:
    if score >= 0.3:
        return "POSITIVE"
    elif score <= -0.3:
        return "NEGATIVE"
    return "NEUTRAL"


def get_sentiment(symbol: str, company_name: str = "") -> dict:
    """
    Fetch news and return sentiment analysis for a stock.

    Returns:
    {
      "symbol": "TCS.NS",
      "overall_score": 0.32,        # -1 to +1
      "overall_label": "POSITIVE",
      "headline_count": 10,
      "breakdown": {"POSITIVE": 6, "NEUTRAL": 3, "NEGATIVE": 1},
      "headlines": [
        {"title": "...", "score": 0.5, "label": "POSITIVE", "published": "..."},
        ...
      ]
    }
    """
    # Look up company name from stocks list if not provided
    if not company_name:
        try:
            from stocks_list import INDIAN_STOCKS
            for s in INDIAN_STOCKS:
                if s["symbol"] == symbol:
                    company_name = s["name"]
                    break
        except Exception:
            pass

    articles = _fetch_rss(symbol, company_name)

    if not articles:
        return {
            "symbol":        symbol,
            "overall_score": 0.0,
            "overall_label": "NEUTRAL",
            "headline_count": 0,
            "breakdown":     {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0},
            "headlines":     [],
            "error":         "Could not fetch news. Check internet connection.",
        }

    scored = []
    breakdown = {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}

    for art in articles:
        score = _score_headline(art["title"])
        label = _label(score)
        breakdown[label] += 1
        scored.append({
            "title":     art["title"],
            "score":     score,
            "label":     label,
            "published": art["published"],
        })

    # Overall score = weighted average (recent headlines matter more)
    if scored:
        scores      = [h["score"] for h in scored]
        # Simple decay: most recent gets highest weight
        weights     = [1 / (i + 1) for i in range(len(scores))]
        total_w     = sum(weights)
        overall     = sum(s * w for s, w in zip(scores, weights)) / total_w
        overall     = round(overall, 3)
    else:
        overall = 0.0

    return {
        "symbol":         symbol,
        "company":        company_name,
        "overall_score":  overall,
        "overall_label":  _label(overall),
        "headline_count": len(scored),
        "breakdown":      breakdown,
        "headlines":      scored[:10],  # return top 10
    }