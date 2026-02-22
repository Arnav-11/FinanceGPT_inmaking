"""
portfolio_optimizer.py  (place in: services/portfolio_optimizer.py)
--------------------------------------------------------------------
Portfolio Optimization using Modern Portfolio Theory (MPT).

Computes:
  1. Maximum Sharpe Ratio portfolio - best risk-adjusted return
  2. Minimum Volatility portfolio   - lowest risk allocation
  3. Equal-weight baseline          - naive comparison
  4. Current portfolio stats        - if using virtual portfolio
  5. Efficient frontier chart data  - for visualization
  6. Correlation matrix             - find diversification opportunities

Math used:
  - Expected return  = mean log return * 252   (annualized)
  - Volatility       = std of daily returns * sqrt(252)
  - Sharpe Ratio     = (return - risk_free) / volatility
  - Monte Carlo simulation (5000 random portfolios) + scipy minimize

No external finance library needed. Uses only numpy + scipy.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Optional

log = logging.getLogger(__name__)

RISK_FREE_RATE   = 0.065   # 6.5% Indian 10-yr govt bond yield
TRADING_DAYS     = 252
N_PORTFOLIOS     = 5000    # Monte Carlo simulations
MIN_HISTORY_DAYS = 120     # Minimum rows needed per stock


def _load_returns(symbols: list) -> pd.DataFrame:
    """Load daily log returns for all symbols from cached CSVs."""
    returns = {}
    for symbol in symbols:
        fp = os.path.join("data_storage", symbol.replace(".", "_") + ".csv")
        if not os.path.exists(fp):
            log.warning(f"No CSV for {symbol} - skipping")
            continue
        try:
            df = pd.read_csv(fp, index_col=0, parse_dates=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            else:
                df.columns = [str(c).strip().split(" ")[0] for c in df.columns]
            if "Close" not in df.columns or len(df) < MIN_HISTORY_DAYS:
                continue
            close = df["Close"].tail(TRADING_DAYS * 3).dropna()
            log_ret = np.log(close / close.shift(1)).dropna()
            if len(log_ret) >= MIN_HISTORY_DAYS:
                returns[symbol] = log_ret
        except Exception as e:
            log.warning(f"Error loading {symbol}: {e}")

    if not returns:
        return pd.DataFrame()
    df_out = pd.DataFrame(returns)
    df_out.dropna(inplace=True)
    return df_out


def _portfolio_stats(weights, mean_returns, cov_matrix):
    """Return (annual_return, annual_vol, sharpe) for given weights."""
    w   = np.array(weights)
    ret = float(np.sum(mean_returns * w) * TRADING_DAYS)
    vol = float(np.sqrt(np.dot(w.T, np.dot(cov_matrix * TRADING_DAYS, w))))
    sr  = (ret - RISK_FREE_RATE) / vol if vol > 0 else 0.0
    return ret, vol, sr


def _monte_carlo(mean_returns, cov_matrix, n=N_PORTFOLIOS):
    """Generate n random portfolios for the efficient frontier."""
    n_stocks = len(mean_returns)
    rng = np.random.default_rng(42)
    results = []
    for _ in range(n):
        w = rng.random(n_stocks)
        w /= w.sum()
        ret, vol, sr = _portfolio_stats(w, mean_returns, cov_matrix)
        results.append({"weights": w.tolist(), "return": ret, "volatility": vol, "sharpe": sr})
    return results


def _optimize(mean_returns, cov_matrix, mode="sharpe"):
    """Use scipy to find exact optimal weights."""
    try:
        from scipy.optimize import minimize
        n  = len(mean_returns)
        x0 = np.ones(n) / n
        bounds      = [(0.02, 0.40)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        if mode == "sharpe":
            fn = lambda w: -_portfolio_stats(w, mean_returns, cov_matrix)[2]
        else:
            fn = lambda w: _portfolio_stats(w, mean_returns, cov_matrix)[1]

        res = minimize(fn, x0, method="SLSQP", bounds=bounds,
                       constraints=constraints, options={"maxiter": 1000})
        if res.success:
            w = res.x
            return w / w.sum()
    except ImportError:
        log.warning("scipy not available - using Monte Carlo best")
    return None


def _fmt_alloc(weights, symbols):
    """Format weights as sorted allocation list."""
    return sorted(
        [{"symbol": s, "weight_pct": round(float(w) * 100, 2), "weight": round(float(w), 4)}
         for s, w in zip(symbols, weights)],
        key=lambda x: x["weight_pct"], reverse=True
    )


def optimize_portfolio(
    symbols:               Optional[list] = None,
    use_virtual_portfolio: bool           = False,
) -> dict:
    """
    Optimize a portfolio of Indian stocks.

    Pass symbols=['TCS.NS','INFY.NS',...] or set use_virtual_portfolio=True
    to use your current virtual portfolio holdings.
    """
    port_holdings   = None
    current_shares  = None
    current_prices  = None

    if use_virtual_portfolio or not symbols:
        try:
            from services.portfolio import _load as load_portfolio
            port = load_portfolio()
            holdings = port.get("holdings", {})
            if not holdings:
                return {"error": "Virtual portfolio is empty. Buy stocks first."}
            symbols        = list(holdings.keys())
            current_shares = {s: h["shares"]    for s, h in holdings.items()}
            current_prices = {s: h["avg_price"] for s, h in holdings.items()}
            port_holdings  = holdings
        except Exception as e:
            return {"error": f"Could not load portfolio: {e}"}

    if len(symbols) < 2:
        return {"error": "Need at least 2 stocks to optimize."}

    returns_df = _load_returns(symbols)
    if returns_df.empty or len(returns_df.columns) < 2:
        return {"error": "Not enough cached data. Run POST /data/download-all first."}

    syms        = list(returns_df.columns)
    missing     = list(set(symbols) - set(syms))
    n           = len(syms)
    mu          = returns_df.mean().values
    cov         = returns_df.cov().values

    # Monte Carlo frontier
    mc = _monte_carlo(mu, cov)
    mc_best_sr  = max(mc, key=lambda x: x["sharpe"])
    mc_min_vol  = min(mc, key=lambda x: x["volatility"])

    # Scipy exact optimization
    w_sharpe = _optimize(mu, cov, mode="sharpe")
    w_minvol = _optimize(mu, cov, mode="min_vol")
    w_sharpe = w_sharpe if w_sharpe is not None else np.array(mc_best_sr["weights"])
    w_minvol = w_minvol if w_minvol is not None else np.array(mc_min_vol["weights"])

    r_sh, v_sh, s_sh = _portfolio_stats(w_sharpe, mu, cov)
    r_mv, v_mv, s_mv = _portfolio_stats(w_minvol, mu, cov)

    w_eq             = np.ones(n) / n
    r_eq, v_eq, s_eq = _portfolio_stats(w_eq, mu, cov)

    # Current portfolio stats (if from virtual portfolio)
    current_stats = None
    if current_shares and current_prices:
        try:
            vals  = {s: current_shares[s] * current_prices[s] for s in syms if s in current_shares}
            total = sum(vals.values())
            if total > 0:
                w_curr = np.array([vals.get(s, 0) / total for s in syms])
                r_c, v_c, s_c = _portfolio_stats(w_curr, mu, cov)
                current_stats = {
                    "allocation":      _fmt_alloc(w_curr, syms),
                    "expected_return": f"{r_c * 100:.2f}%",
                    "volatility":      f"{v_c * 100:.2f}%",
                    "sharpe_ratio":    round(s_c, 3),
                    "note": "Based on your current virtual portfolio"
                }
        except Exception as e:
            log.warning(f"Could not compute current portfolio stats: {e}")

    # Per-stock individual stats
    individual = sorted([
        {
            "symbol":           s,
            "annual_return":    f"{float(returns_df[s].mean() * TRADING_DAYS) * 100:.2f}%",
            "annual_volatility": f"{float(returns_df[s].std() * np.sqrt(TRADING_DAYS)) * 100:.2f}%",
            "sharpe":           round(
                (float(returns_df[s].mean() * TRADING_DAYS) - RISK_FREE_RATE) /
                max(float(returns_df[s].std() * np.sqrt(TRADING_DAYS)), 1e-9), 3
            ),
        }
        for s in syms
    ], key=lambda x: x["sharpe"], reverse=True)

    # Top and low correlations for diversification insight
    corr = returns_df.corr().round(3)
    corr_pairs = [
        {"stock1": s1, "stock2": s2, "correlation": float(corr.loc[s1, s2])}
        for i, s1 in enumerate(syms)
        for j, s2 in enumerate(syms)
        if i < j
    ]
    corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    # Frontier sample for charting (100 points)
    step = max(1, N_PORTFOLIOS // 100)
    frontier = [
        {"return": round(p["return"] * 100, 3),
         "volatility": round(p["volatility"] * 100, 3),
         "sharpe": round(p["sharpe"], 3)}
        for p in mc[::step]
    ]

    return {
        "symbols_used":    syms,
        "symbols_missing": missing,
        "data_period_days": len(returns_df),

        "max_sharpe_portfolio": {
            "allocation":      _fmt_alloc(w_sharpe, syms),
            "expected_return": f"{r_sh * 100:.2f}%",
            "volatility":      f"{v_sh * 100:.2f}%",
            "sharpe_ratio":    round(s_sh, 3),
            "description":     "Best risk-adjusted return (maximize Sharpe ratio)",
        },
        "min_volatility_portfolio": {
            "allocation":      _fmt_alloc(w_minvol, syms),
            "expected_return": f"{r_mv * 100:.2f}%",
            "volatility":      f"{v_mv * 100:.2f}%",
            "sharpe_ratio":    round(s_mv, 3),
            "description":     "Lowest risk allocation (minimize volatility)",
        },
        "equal_weight_portfolio": {
            "allocation":      _fmt_alloc(w_eq, syms),
            "expected_return": f"{r_eq * 100:.2f}%",
            "volatility":      f"{v_eq * 100:.2f}%",
            "sharpe_ratio":    round(s_eq, 3),
            "description":     "Naive equal allocation (baseline comparison)",
        },
        "current_portfolio":        current_stats,
        "individual_stock_stats":   individual,
        "top_correlations":         corr_pairs[:10],
        "low_correlations":         corr_pairs[-10:],
        "frontier_chart_data":      frontier,
        "risk_free_rate":           f"{RISK_FREE_RATE * 100:.1f}%",
        "note": "Weights capped 2%-40% per stock. Past performance ≠ future results.",
    }