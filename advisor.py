"""
advisor.py  (place in: services/advisor.py)
--------------------------------------------
Conversational Portfolio Advisor powered by Claude AI.

Understands natural language questions, fetches real data from
your own FinanceGPT modules, and returns data-backed answers.

Example questions:
  "Is TCS a good buy right now?"
  "What should I do with my portfolio?"
  "Which IT stocks look bullish?"
  "Compare HDFC Bank and ICICI Bank"
  "Optimize my portfolio allocation"

Setup:
  1. pip install anthropic
  2. Set ANTHROPIC_API_KEY in your .env or environment
  3. Add `from services.advisor import chat` and add routes to main.py
"""

import os
import json
import logging
from typing import Optional

log = logging.getLogger(__name__)

# In-memory conversation store (session_id -> message history)
# For production: move to Redis or a DB
_sessions: dict = {}


def _client():
    """Get Anthropic client. Raises clear error if not configured."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("Run: pip install anthropic")
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Add to your .env: ANTHROPIC_API_KEY=sk-ant-..."
        )
    return anthropic.Anthropic(api_key=key)


# ── Data helpers ──────────────────────────────────────────────────────────────

def _safe(fn, *args, **kwargs):
    """Call a function, return error dict on failure."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return {"error": str(e)}


def _analysis(symbol):
    from services.analysis import analyze_stock
    return analyze_stock(symbol)

def _portfolio():
    from services.portfolio import get_portfolio
    return get_portfolio()

def _predict(symbol):
    from ml_models.predict import predict
    return predict(symbol)

def _screener(filters):
    from services.screener import run_screener
    return run_screener(**filters, limit=8)

def _sector():
    from services.sector_analysis import get_sector_heatmap
    return get_sector_heatmap()

def _optimizer(symbols):
    from services.portfolio_optimizer import optimize_portfolio
    return optimize_portfolio(symbols=symbols)


# ── Intent detection ──────────────────────────────────────────────────────────

SYMBOL_MAP = {
    "tcs": "TCS.NS", "infosys": "INFY.NS", "wipro": "WIPRO.NS",
    "hcltech": "HCLTECH.NS", "reliance": "RELIANCE.NS",
    "hdfc bank": "HDFCBANK.NS", "hdfcbank": "HDFCBANK.NS",
    "icici bank": "ICICIBANK.NS", "icicibank": "ICICIBANK.NS",
    "sbi": "SBIN.NS", "state bank": "SBIN.NS",
    "bajaj finance": "BAJFINANCE.NS", "bajfinance": "BAJFINANCE.NS",
    "asian paints": "ASIANPAINT.NS", "maruti": "MARUTI.NS",
    "larsen": "LT.NS", "l&t": "LT.NS", "titan": "TITAN.NS",
    "kotak": "KOTAKBANK.NS", "axis bank": "AXISBANK.NS",
    "sun pharma": "SUNPHARMA.NS", "itc": "ITC.NS",
    "ongc": "ONGC.NS", "ntpc": "NTPC.NS", "tatasteel": "TATASTEEL.NS",
    "tata motors": "TATAMOTORS.NS", "tata steel": "TATASTEEL.NS",
}

def _detect_intent(question: str, history: list) -> dict:
    """
    Use a fast Claude call to detect intent and extract symbols.
    Falls back to safe defaults if it fails.
    """
    client = _client()
    prompt = f"""Analyze this finance question and return ONLY a JSON object.

Question: "{question}"

Return JSON with:
{{
  "intent": one of ["stock_analysis","portfolio_review","comparison","screener","sector","prediction","optimizer","general"],
  "symbols": [list of stock symbols mentioned, add .NS suffix for Indian stocks],
  "screener_filters": {{only if intent=screener, e.g. {{"rsi_below": 35, "macd_bullish": true}}}}
}}

Common Indian stock symbols: TCS.NS, INFY.NS, RELIANCE.NS, HDFCBANK.NS, ICICIBANK.NS,
SBIN.NS, BAJFINANCE.NS, WIPRO.NS, HCLTECH.NS, AXISBANK.NS, KOTAKBANK.NS, MARUTI.NS

Only output the JSON, nothing else."""

    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        raw  = resp.content[0].text.strip().replace("```json","").replace("```","").strip()
        data = json.loads(raw)
        # Normalize symbols
        data["symbols"] = [
            s if "." in s else s + ".NS"
            for s in data.get("symbols", [])
        ]
        return data
    except Exception as e:
        log.warning(f"Intent detection failed: {e}")
        # Simple fallback
        q_lower = question.lower()
        symbols = []
        for name, sym in SYMBOL_MAP.items():
            if name in q_lower:
                symbols.append(sym)
        return {"intent": "general", "symbols": list(set(symbols)), "screener_filters": {}}


def _gather_data(intent_data: dict) -> dict:
    """Gather real data from FinanceGPT modules based on detected intent."""
    intent  = intent_data.get("intent", "general")
    symbols = intent_data.get("symbols", [])
    gathered = {}

    if intent == "stock_analysis" and symbols:
        gathered["analyses"]    = [_safe(_analysis, s) for s in symbols[:3]]
        gathered["predictions"] = [r for s in symbols[:3]
                                   if "error" not in (r := _safe(_predict, s))]

    elif intent == "portfolio_review":
        port = _safe(_portfolio)
        gathered["portfolio"] = port
        if isinstance(port, dict) and "holdings" in port:
            h_syms = [h["symbol"] for h in port["holdings"][:5]]
            gathered["holding_analyses"] = [_safe(_analysis, s) for s in h_syms]

    elif intent == "comparison" and len(symbols) >= 2:
        gathered["analyses"]    = [_safe(_analysis, s) for s in symbols[:4]]
        gathered["predictions"] = [r for s in symbols[:4]
                                   if "error" not in (r := _safe(_predict, s))]

    elif intent == "screener":
        filters = intent_data.get("screener_filters", {})
        gathered["screener_results"] = _safe(_screener, filters)

    elif intent == "sector":
        gathered["sector_data"] = _safe(_sector)

    elif intent == "prediction" and symbols:
        gathered["predictions"] = [_safe(_predict, s) for s in symbols[:3]]
        gathered["analyses"]    = [_safe(_analysis, s) for s in symbols[:3]]

    elif intent == "optimizer":
        if symbols:
            gathered["optimization"] = _safe(_optimizer, symbols)
        else:
            port = _safe(_portfolio)
            gathered["portfolio"] = port
            if isinstance(port, dict) and "holdings" in port and port["holdings"]:
                port_syms = [h["symbol"] for h in port["holdings"]]
                gathered["optimization"] = _safe(_optimizer, port_syms)

    return gathered


# ── Main chat function ────────────────────────────────────────────────────────

def chat(question: str, session_id: str = "default") -> dict:
    """
    Ask the AI advisor a finance question.

    Parameters
    ----------
    question   : Natural language finance question
    session_id : Session ID for multi-turn conversation memory

    Returns
    -------
    {
      "answer":      str   - Claude's natural language response,
      "session_id":  str   - session ID for follow-up questions,
      "intent":      str   - detected question intent,
      "data_used":   list  - what data sources were consulted,
      "turn_number": int   - conversation turn count
    }
    """
    client = _client()

    if session_id not in _sessions:
        _sessions[session_id] = []
    history = _sessions[session_id]

    # Detect intent and gather data
    intent_data  = _detect_intent(question, history)
    gathered     = _gather_data(intent_data)

    # Build context string from gathered data
    data_ctx = ""
    if gathered:
        data_ctx = (
            "\n\nREAL-TIME MARKET DATA from FinanceGPT system:\n"
            + json.dumps(gathered, indent=2, default=str)[:6000]
            + "\n\nUse this data to ground your response in real numbers.\n"
        )

    system = f"""You are FinanceGPT, an expert AI financial advisor for Indian stock markets (NSE/BSE).

You have access to real-time technical analysis, ML predictions, portfolio data and sector analysis.

Your style:
- Data-driven: always cite RSI, price, score numbers from the provided data
- Clear and practical: give specific actionable advice
- Balanced: mention both opportunities AND risks for every recommendation
- Plain language: explain indicators simply (e.g. "RSI of 28 means the stock is oversold")
- Honest: never guarantee returns, always include a short risk reminder

Format guidelines:
- Use short paragraphs, not bullet walls
- Bold key numbers and verdicts
- End investment-related responses with a 1-line risk disclaimer
{data_ctx}"""

    messages = history + [{"role": "user", "content": question}]

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1500,
        system=system,
        messages=messages,
    )
    answer = response.content[0].text

    # Update history (keep last 20 messages = 10 turns)
    history.append({"role": "user",      "content": question})
    history.append({"role": "assistant", "content": answer})
    _sessions[session_id] = history[-20:]

    return {
        "answer":      answer,
        "session_id":  session_id,
        "intent":      intent_data.get("intent", "general"),
        "data_used":   list(gathered.keys()),
        "turn_number": len(_sessions[session_id]) // 2,
    }


def clear_session(session_id: str = "default") -> dict:
    """Clear conversation history for a session."""
    _sessions.pop(session_id, None)
    return {"message": f"Session '{session_id}' cleared"}


def get_history(session_id: str = "default") -> list:
    """Return conversation history for a session."""
    return _sessions.get(session_id, [])