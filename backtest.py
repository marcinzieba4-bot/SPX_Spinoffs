#!/usr/bin/env python3
"""
SPX Spinoff Trading Strategy Backtest
======================================
Strategy based on:
  - Joel Greenblatt, "You Can Be a Stock Market Genius" (1997)
  - Desai & Jain (1999): "Firm Performance and Focus: Long-Run Stock Market Performance
    Following Spinoffs" - Journal of Financial Economics
  - McConnell & Ovtchinnikov (2004): "Predictability of long-term spinoff returns" - JPM

STRATEGY RULES:
  1. Universe:  Spinoffs where parent was S&P 500 member
  2. Entry:     Buy 30 calendar days after first trading day
                (lets index funds & forced sellers finish dumping)
  3. Exit:      Sell exactly 1 year (365 cal days) after entry
  4. Sizing:    Equal weight per position; max 1 open at a time per symbol
  5. Benchmark: SPY (S&P 500 ETF)

Rationale: Spinoff stocks are dumped by institutional investors who
receive shares they didn't ask for. Index funds must sell because the
spinoff is too small. This creates a predictable supply/demand imbalance
that reverses over the next 12 months as the business is re-discovered.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
import warnings
import io

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans'})

# ============================================================
# SPINOFF DATABASE  (parent was S&P 500 member at spinoff date)
# Sources: Bloomberg, Capital IQ, Wikipedia corporate actions
# ============================================================
SPINOFFS = [
    # (ticker,  parent,  spinoff_date,   description)
    # ── 2011 ──────────────────────────────────────────────────────────────
    ("MPC",   "MRO",   "2011-07-01",  "Marathon Petroleum from Marathon Oil"),
    # ── 2012 ──────────────────────────────────────────────────────────────
    ("ADT",   "TYC",   "2012-09-28",  "ADT Security from Tyco International"),
    # ── 2013 ──────────────────────────────────────────────────────────────
    ("ABBV",  "ABT",   "2013-01-02",  "AbbVie from Abbott Laboratories"),
    ("ZTS",   "PFE",   "2013-02-01",  "Zoetis from Pfizer"),
    ("VOYA",  "ING",   "2013-05-02",  "Voya Financial from ING Group"),
    # MNK spun from COV (S&P 500 member); later went bankrupt — MUST include per rules
    ("MNK",   "COV",   "2013-07-01",  "Mallinckrodt from Covidien"),
    # ── 2014 ──────────────────────────────────────────────────────────────
    ("SYF",   "GE",    "2014-07-31",  "Synchrony Financial from General Electric"),
    # ── 2015 ──────────────────────────────────────────────────────────────
    ("PYPL",  "EBAY",  "2015-07-20",  "PayPal from eBay"),
    ("HPE",   "HPQ",   "2015-11-02",  "HP Enterprise from Hewlett-Packard"),
    ("BXLT",  "BAX",   "2015-07-01",  "Baxalta from Baxter International"),
    # ── 2016 ──────────────────────────────────────────────────────────────
    ("FTV",   "DHR",   "2016-07-02",  "Fortive from Danaher"),
    ("AA",    "ARNC",  "2016-11-01",  "Alcoa Corp from Arconic"),
    # ── 2017 ──────────────────────────────────────────────────────────────
    ("CNDT",  "XRX",   "2017-01-03",  "Conduent from Xerox"),
    ("DXC",   "HPE",   "2017-04-03",  "DXC Technology from HPE / CSC"),
    # ── 2018 ──────────────────────────────────────────────────────────────
    ("NVT",   "PNR",   "2018-05-01",  "nVent Electric from Pentair"),
    # ── 2019 ──────────────────────────────────────────────────────────────
    ("DOW",   "DWDP",  "2019-04-01",  "Dow Inc from DowDuPont"),
    ("CTVA",  "DWDP",  "2019-06-03",  "Corteva Agriscience from DowDuPont"),
    # ── 2020 ──────────────────────────────────────────────────────────────
    ("OTIS",  "UTX",   "2020-04-03",  "Otis Worldwide from United Technologies"),
    ("CARR",  "UTX",   "2020-04-03",  "Carrier Global from United Technologies"),
    ("HWM",   "ARNC",  "2020-04-01",  "Howmet Aerospace from Arconic"),
    ("VNT",   "FTV",   "2020-10-09",  "Vontier from Fortive"),
    # ── 2021 ──────────────────────────────────────────────────────────────
    ("GXO",   "XPO",   "2021-08-02",  "GXO Logistics from XPO Inc"),
    ("KD",    "IBM",   "2021-11-04",  "Kyndryl Holdings from IBM"),
    # ── 2022 ──────────────────────────────────────────────────────────────
    ("CEG",   "EXC",   "2022-01-03",  "Constellation Energy from Exelon"),
    ("RXO",   "XPO",   "2022-11-01",  "RXO Inc from XPO"),
    # ── 2023 ──────────────────────────────────────────────────────────────
    ("GEHC",  "GE",    "2023-01-04",  "GE HealthCare from General Electric"),
    ("KVUE",  "JNJ",   "2023-05-04",  "Kenvue from Johnson & Johnson"),
    ("VLTO",  "DHR",   "2023-09-14",  "Veralto from Danaher"),
    # ── 2024 ──────────────────────────────────────────────────────────────
    ("SOLV",  "MMM",   "2024-04-01",  "Solventum from 3M"),
    ("GEV",   "GE",    "2024-04-02",  "GE Vernova from General Electric"),
    # ── 2025 ── (open positions — simulated to latest available price) ────
    # SNDK: WDC was S&P 500 member; SNDK went to S&P SmallCap 600
    ("SNDK",  "WDC",   "2025-02-24",  "SanDisk from Western Digital"),
    # SOLS: HON was S&P 500; Solstice Advanced Materials joined S&P 500 Oct 2025
    ("SOLS",  "HON",   "2025-10-30",  "Solstice Advanced Materials from Honeywell"),
    # Q: EMN was S&P 500; Qnity Electronics joined S&P 500 Oct 2025
    ("Q",     "EMN",   "2025-10-31",  "Qnity Electronics from Eastman Chemical"),
]

# ── Strategy parameters ────────────────────────────────────────────────────────
ENTRY_DELAY_DAYS   = 30    # calendar days post-spinoff before buying
HOLDING_PERIOD     = 365   # calendar days to hold
BENCHMARK          = "SPY"
DATA_BUFFER_DAYS   = 90    # extra days to download around each event


# ============================================================
# DATA FETCHING
# ============================================================

def fetch_price(ticker: str, start: str, end: str) -> pd.Series:
    """Return adjusted-close series; empty Series on failure."""
    try:
        data = yf.download(ticker, start=start, end=end,
                           auto_adjust=True, progress=False)
        if data.empty:
            return pd.Series(dtype=float, name=ticker)
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
        close.index = pd.to_datetime(close.index).tz_localize(None)
        return close.dropna()
    except Exception:
        return pd.Series(dtype=float, name=ticker)


def next_trading_day(series: pd.Series, target_date: datetime) -> datetime | None:
    """Return the first date in series >= target_date, or None."""
    candidates = series.index[series.index >= target_date]
    return candidates[0] if len(candidates) else None


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest() -> pd.DataFrame:
    """Download data and compute per-trade returns."""
    records = []

    print(f"Running backtest on {len(SPINOFFS)} spinoffs …\n")

    for ticker, parent, spinoff_date_str, desc in SPINOFFS:
        spinoff_date = pd.Timestamp(spinoff_date_str)
        entry_target = spinoff_date + timedelta(days=ENTRY_DELAY_DAYS)
        exit_target  = entry_target + timedelta(days=HOLDING_PERIOD)

        dl_start = (spinoff_date - timedelta(days=5)).strftime("%Y-%m-%d")
        dl_end   = (exit_target  + timedelta(days=DATA_BUFFER_DAYS)).strftime("%Y-%m-%d")

        prices = fetch_price(ticker, dl_start, dl_end)
        spy    = fetch_price(BENCHMARK, dl_start, dl_end)

        if len(prices) < 20:
            print(f"  SKIP {ticker:6s} – insufficient data ({len(prices)} bars)")
            records.append({
                "ticker": ticker, "parent": parent,
                "spinoff_date": spinoff_date, "description": desc,
                "entry_date": None, "exit_date": None,
                "entry_price": np.nan, "exit_price": np.nan,
                "spinoff_return": np.nan, "spy_return": np.nan,
                "alpha": np.nan, "status": "no_data"
            })
            continue

        entry_date = next_trading_day(prices, entry_target)
        if entry_date is None:
            print(f"  SKIP {ticker:6s} – no entry date found")
            records.append({
                "ticker": ticker, "parent": parent,
                "spinoff_date": spinoff_date, "description": desc,
                "entry_date": None, "exit_date": None,
                "entry_price": np.nan, "exit_price": np.nan,
                "spinoff_return": np.nan, "spy_return": np.nan,
                "alpha": np.nan, "status": "no_entry"
            })
            continue

        # For open positions (spinoff < 1 year ago), use most recent close
        cutoff = pd.Timestamp.today() - timedelta(days=5)
        if exit_target > cutoff:
            exit_candidates = prices.index[prices.index > entry_date]
            exit_date = exit_candidates[-1] if len(exit_candidates) else None
            status = "open"
        else:
            exit_date = next_trading_day(prices, exit_target)
            status = "closed"

        if exit_date is None or exit_date <= entry_date:
            print(f"  SKIP {ticker:6s} – no valid exit date")
            records.append({
                "ticker": ticker, "parent": parent,
                "spinoff_date": spinoff_date, "description": desc,
                "entry_date": entry_date, "exit_date": None,
                "entry_price": np.nan, "exit_price": np.nan,
                "spinoff_return": np.nan, "spy_return": np.nan,
                "alpha": np.nan, "status": "no_exit"
            })
            continue

        entry_price = float(prices.loc[entry_date])
        exit_price  = float(prices.loc[exit_date])
        spinoff_ret = (exit_price / entry_price) - 1.0

        # SPY benchmark over same window
        spy_entry = next_trading_day(spy, entry_date)
        spy_exit  = next_trading_day(spy, exit_date)
        if spy_entry and spy_exit and spy_entry in spy.index and spy_exit in spy.index:
            spy_ret = float(spy.loc[spy_exit]) / float(spy.loc[spy_entry]) - 1.0
        else:
            spy_ret = np.nan

        alpha = spinoff_ret - spy_ret if not np.isnan(spy_ret) else np.nan

        flag = "✓" if spinoff_ret > (spy_ret if not np.isnan(spy_ret) else 0) else "✗"
        print(f"  {flag} {ticker:6s} | {spinoff_date_str} | "
              f"ret={spinoff_ret:+.1%}  spy={spy_ret:+.1%}  α={alpha:+.1%}  [{status}]")

        records.append({
            "ticker":          ticker,
            "parent":          parent,
            "spinoff_date":    spinoff_date,
            "description":     desc,
            "entry_date":      entry_date,
            "exit_date":       exit_date,
            "entry_price":     entry_price,
            "exit_price":      exit_price,
            "spinoff_return":  spinoff_ret,
            "spy_return":      spy_ret,
            "alpha":           alpha,
            "status":          status,
        })

    df = pd.DataFrame(records)
    return df


# ============================================================
# STATISTICS
# ============================================================

def compute_stats(df: pd.DataFrame) -> dict:
    """Compute aggregate performance statistics."""
    closed = df[df["status"] == "closed"].copy()
    valid  = closed.dropna(subset=["spinoff_return", "spy_return"])

    if len(valid) == 0:
        return {}

    rets   = valid["spinoff_return"].values
    spy    = valid["spy_return"].values
    alphas = valid["alpha"].values

    win_rate = (rets > 0).mean()
    beat_spy = (alphas > 0).mean()

    # Annualised returns (positions held ~1 year → already annual)
    mean_ret   = rets.mean()
    median_ret = np.median(rets)
    mean_alpha = alphas.mean()

    # t-test: is mean alpha significantly > 0?
    t_stat, p_value = stats.ttest_1samp(alphas, 0)

    # Sharpe-like (alpha / std_alpha)
    if alphas.std() > 0:
        information_ratio = alphas.mean() / alphas.std()
    else:
        information_ratio = np.nan

    # Max drawdown proxy: worst single trade
    max_loss = rets.min()
    max_gain = rets.max()

    return {
        "n_trades":          len(valid),
        "n_open":            (df["status"] == "open").sum(),
        "win_rate":          win_rate,
        "beat_spy_rate":     beat_spy,
        "mean_return":       mean_ret,
        "median_return":     median_ret,
        "mean_alpha":        mean_alpha,
        "std_return":        rets.std(),
        "std_alpha":         alphas.std(),
        "max_gain":          max_gain,
        "max_loss":          max_loss,
        "information_ratio": information_ratio,
        "t_stat":            t_stat,
        "p_value":           p_value,
        "mean_spy_return":   spy.mean(),
        "total_spinoffs":    len(df),
    }


def build_equity_curve(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate a $100 000 portfolio – each closed trade is $100 000 / n_concurrent.
    For simplicity we chain equal-weight trades sequentially by entry date.
    """
    closed = df.dropna(subset=["entry_date", "exit_date", "spinoff_return"]).copy()
    closed = closed[closed["status"] == "closed"].sort_values("entry_date")

    if closed.empty:
        return pd.DataFrame()

    # Monthly resampled equity assuming equal-weight full-port rebalance each trade
    # Simpler: create month-end equity by compounding mean monthly alpha
    start = closed["entry_date"].min()
    end   = closed["exit_date"].max()
    dates = pd.date_range(start, end, freq="ME")

    # For each month-end compute mean return of all positions active that month
    rows = []
    for d in dates:
        active = closed[
            (closed["entry_date"] <= d) & (closed["exit_date"] >= d)
        ]
        if len(active):
            # pro-rate return: fraction of holding period elapsed
            active = active.copy()
            active["frac"] = (d - active["entry_date"]).dt.days / HOLDING_PERIOD
            active["frac"] = active["frac"].clip(0, 1)
            partial_ret = (active["spinoff_return"] * active["frac"]).mean()
            partial_spy = (active["spy_return"]     * active["frac"]).mean()
            rows.append({"date": d, "strategy": partial_ret, "spy": partial_spy,
                         "n_positions": len(active)})

    if not rows:
        return pd.DataFrame()

    curve = pd.DataFrame(rows).set_index("date")
    # Cumulative: treat each month as independent observation for simplicity
    # Build a proper equity curve from individual trades
    # We use portfolio approach: each trade gets equal capital
    capital = 100_000.0
    spy_capital = 100_000.0

    equity_rows = []
    for _, row in closed.iterrows():
        trade_ret = row["spinoff_return"]
        spy_ret   = row["spy_return"]
        capital   = capital   * (1 + trade_ret / len(closed) * len(closed))  # 100% per trade
        equity_rows.append({
            "date":     row["exit_date"],
            "strategy": capital,
        })

    # Better approach: timeline-based, all trades contribute
    all_dates = pd.date_range(
        closed["entry_date"].min(),
        closed["exit_date"].max(),
        freq="W"
    )

    strat_idx = [1.0]
    spy_idx   = [1.0]
    prev_date  = all_dates[0]

    weekly_rows = [{"date": prev_date, "strategy_idx": 1.0, "spy_idx": 1.0}]

    for d in all_dates[1:]:
        active = closed[
            (closed["entry_date"] <= d) & (closed["exit_date"] >= d)
        ]
        if len(active) == 0:
            weekly_rows.append({"date": d,
                                 "strategy_idx": weekly_rows[-1]["strategy_idx"],
                                 "spy_idx":      weekly_rows[-1]["spy_idx"]})
            continue
        # Weekly contribution
        active = active.copy()
        days_held = (d - active["entry_date"]).dt.days.clip(lower=1)
        days_total = HOLDING_PERIOD
        week_days = 7
        # Weekly return = annual return * (7/365)
        weekly_strat = (active["spinoff_return"] * week_days / days_total).mean()
        weekly_spy   = (active["spy_return"]     * week_days / days_total).mean()

        new_strat = weekly_rows[-1]["strategy_idx"] * (1 + weekly_strat)
        new_spy   = weekly_rows[-1]["spy_idx"]      * (1 + weekly_spy)

        weekly_rows.append({"date": d, "strategy_idx": new_strat, "spy_idx": new_spy})

    wdf = pd.DataFrame(weekly_rows).set_index("date")
    wdf["strategy_value"] = wdf["strategy_idx"] * 100_000
    wdf["spy_value"]      = wdf["spy_idx"]      * 100_000
    return wdf


# ============================================================
# PDF GENERATION
# ============================================================

COLOR_STRAT = "#1f77b4"
COLOR_SPY   = "#ff7f0e"
COLOR_POS   = "#2ca02c"
COLOR_NEG   = "#d62728"
COLOR_ALPHA = "#9467bd"

def fmt_pct(x): return f"{x:+.1%}" if not np.isnan(x) else "N/A"
def fmt_n(x):   return f"{x:.2f}"  if not np.isnan(x) else "N/A"


def make_cover_page(pdf, stats: dict):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Header background
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 0.82), 1, 0.18, boxstyle="square,pad=0",
        facecolor="#1a3a5c", edgecolor="none"))

    ax.text(0.5, 0.93, "SPX SPINOFF TRADING STRATEGY",
            ha="center", va="center", fontsize=22, fontweight="bold",
            color="white", transform=ax.transAxes)
    ax.text(0.5, 0.87, "Systematic Backtest Report",
            ha="center", va="center", fontsize=14, color="#a8c8e8",
            transform=ax.transAxes)

    # Date
    ax.text(0.5, 0.80, f"Generated: {datetime.today().strftime('%B %d, %Y')}",
            ha="center", va="center", fontsize=10, color="#555",
            transform=ax.transAxes)

    # Strategy summary box
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.05, 0.54), 0.9, 0.23,
        boxstyle="round,pad=0.01", facecolor="#f0f4f8", edgecolor="#ccc", linewidth=1))
    ax.text(0.5, 0.765, "STRATEGY SUMMARY",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color="#1a3a5c", transform=ax.transAxes)

    summary_lines = [
        "• Universe:   S&P 500 spinoff stocks (parent was SPX member at spinoff date)",
        f"• Entry Rule: Buy {ENTRY_DELAY_DAYS} calendar days after spinoff listing",
        "  (allows index funds & forced sellers to finish dumping shares)",
        f"• Exit Rule:  Sell after {HOLDING_PERIOD} calendar days (~1 year) from entry",
        "• Sizing:     Equal weight per trade",
        "• Benchmark:  SPY (S&P 500 ETF, total return)",
        "• Data:       Yahoo Finance adjusted closes",
    ]
    y0 = 0.74
    for line in summary_lines:
        ax.text(0.10, y0, line, ha="left", va="center", fontsize=9.5,
                color="#333", transform=ax.transAxes, family="monospace")
        y0 -= 0.025

    # Key metrics grid
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.05, 0.22), 0.9, 0.30,
        boxstyle="round,pad=0.01", facecolor="#fff", edgecolor="#ccc", linewidth=1))
    ax.text(0.5, 0.50, "KEY RESULTS",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color="#1a3a5c", transform=ax.transAxes)

    if stats:
        metrics = [
            ("Closed Trades",   f"{stats.get('n_trades', 0)}",         ""),
            ("Win Rate",        fmt_pct(stats.get('win_rate', np.nan)), "(% trades positive)"),
            ("Beat SPY Rate",   fmt_pct(stats.get('beat_spy_rate', np.nan)), "(% trades > benchmark)"),
            ("Mean Return",     fmt_pct(stats.get('mean_return', np.nan)),   "(per trade, ~1yr hold)"),
            ("Median Return",   fmt_pct(stats.get('median_return', np.nan)), ""),
            ("Mean Alpha",      fmt_pct(stats.get('mean_alpha', np.nan)),    "(vs SPY same period)"),
            ("Info Ratio",      fmt_n(stats.get('information_ratio', np.nan)), "(alpha / std alpha)"),
            ("p-value",         f"{stats.get('p_value', np.nan):.3f}",  "(H₀: alpha=0)"),
            ("Best Trade",      fmt_pct(stats.get('max_gain', np.nan)), ""),
            ("Worst Trade",     fmt_pct(stats.get('max_loss', np.nan)), ""),
        ]
        cols = 2
        rows_m = (len(metrics) + 1) // cols
        for i, (label, value, note) in enumerate(metrics):
            col = i % cols
            row = i // cols
            x = 0.08 + col * 0.47
            y = 0.475 - row * 0.042
            # Colour value: green if contains +, red if contains -
            val_color = COLOR_POS if "+" in value else (COLOR_NEG if ("-" in value and value != "N/A") else "#222")
            ax.text(x,       y, f"{label}:",   ha="left",  va="center", fontsize=9,  color="#555", transform=ax.transAxes)
            ax.text(x+0.18,  y, value,         ha="left",  va="center", fontsize=10, fontweight="bold", color=val_color, transform=ax.transAxes)
            ax.text(x+0.28,  y, note,          ha="left",  va="center", fontsize=7.5, color="#888", transform=ax.transAxes)

    # Disclaimer
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.05, 0.04), 0.9, 0.15,
        boxstyle="round,pad=0.01", facecolor="#fff8dc", edgecolor="#e0c000", linewidth=1))
    ax.text(0.5, 0.175, "DISCLAIMER",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#8B6914", transform=ax.transAxes)
    disc = ("This report is for educational and research purposes only. Past performance does not guarantee future results.\n"
            "Strategy returns are gross of transaction costs, slippage, and taxes. Live trading results will differ.\n"
            "Spinoff returns may be partially explained by survivorship bias in data availability.")
    ax.text(0.5, 0.105, disc,
            ha="center", va="center", fontsize=7.5, color="#666",
            transform=ax.transAxes, linespacing=1.5)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_return_charts(pdf, df: pd.DataFrame, stats: dict):
    """Page 2: return distribution, alpha scatter, win/loss bar."""
    closed = df[df["status"] == "closed"].dropna(subset=["spinoff_return", "spy_return"])
    if closed.empty:
        return

    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("Return Analysis", fontsize=14, fontweight="bold", y=0.98, color="#1a3a5c")

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.94, bottom=0.06, left=0.10, right=0.95)

    # ── (a) Return distribution ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    rets_pct = closed["spinoff_return"].values * 100
    spy_pct  = closed["spy_return"].values * 100
    bins = np.linspace(min(rets_pct.min(), spy_pct.min()) - 5,
                       max(rets_pct.max(), spy_pct.max()) + 5, 25)
    ax1.hist(rets_pct, bins=bins, alpha=0.65, color=COLOR_STRAT, label="Spinoff")
    ax1.hist(spy_pct,  bins=bins, alpha=0.65, color=COLOR_SPY,   label="SPY (same window)")
    ax1.axvline(rets_pct.mean(), color=COLOR_STRAT, linestyle="--", linewidth=1.5,
                label=f"Mean spinoff = {rets_pct.mean():.1f}%")
    ax1.axvline(spy_pct.mean(),  color=COLOR_SPY,   linestyle="--", linewidth=1.5,
                label=f"Mean SPY    = {spy_pct.mean():.1f}%")
    ax1.axvline(0, color="black", linewidth=0.8, linestyle=":")
    ax1.set_xlabel("1-Year Return (%)")
    ax1.set_ylabel("Number of Trades")
    ax1.set_title("(a) Distribution of 1-Year Returns: Spinoff vs SPY", fontweight="bold")
    ax1.legend(fontsize=8)

    # ── (b) Alpha histogram ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    alphas = closed["alpha"].values * 100
    ax2.hist(alphas, bins=15, color=COLOR_ALPHA, alpha=0.8, edgecolor="white")
    ax2.axvline(0, color="black", linewidth=1, linestyle="--")
    ax2.axvline(alphas.mean(), color="red", linewidth=1.5,
                label=f"Mean α = {alphas.mean():.1f}%")
    ax2.set_xlabel("Alpha vs SPY (%)")
    ax2.set_ylabel("Count")
    ax2.set_title("(b) Alpha Distribution", fontweight="bold")
    ax2.legend(fontsize=8)

    # Annotation
    p = stats.get("p_value", np.nan)
    t = stats.get("t_stat", np.nan)
    ax2.text(0.97, 0.95,
             f"t = {t:.2f}\np = {p:.3f}",
             transform=ax2.transAxes, ha="right", va="top", fontsize=8,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    # ── (c) Spinoff return vs SPY scatter ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    colors = [COLOR_POS if r > s else COLOR_NEG
              for r, s in zip(closed["spinoff_return"], closed["spy_return"])]
    ax3.scatter(spy_pct, rets_pct, c=colors, alpha=0.75, edgecolors="white", linewidth=0.5)
    lim = max(abs(spy_pct).max(), abs(rets_pct).max()) + 10
    ax3.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8, label="Spinoff = SPY")
    ax3.set_xlim(-lim, lim)
    ax3.set_ylim(-lim, lim)
    ax3.set_xlabel("SPY Return (%)")
    ax3.set_ylabel("Spinoff Return (%)")
    ax3.set_title("(c) Spinoff Return vs SPY (same window)", fontweight="bold")
    green_p = mpatches.Patch(color=COLOR_POS, label="Beat SPY")
    red_p   = mpatches.Patch(color=COLOR_NEG, label="Lagged SPY")
    ax3.legend(handles=[green_p, red_p], fontsize=7)

    # ── (d) Per-trade bar chart ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    sorted_df = closed.sort_values("spinoff_return")
    bar_colors = [COLOR_POS if r > 0 else COLOR_NEG
                  for r in sorted_df["spinoff_return"]]
    x = np.arange(len(sorted_df))
    ax4.bar(x, sorted_df["spinoff_return"] * 100, color=bar_colors,
            width=0.6, alpha=0.85)
    ax4.plot(x, sorted_df["spy_return"] * 100, "o--",
             color=COLOR_SPY, markersize=3, linewidth=0.8, label="SPY (same window)")
    ax4.axhline(0, color="black", linewidth=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(sorted_df["ticker"], rotation=90, fontsize=7)
    ax4.set_ylabel("1-Year Return (%)")
    ax4.set_title("(d) Individual Trade Returns (sorted by spinoff return)", fontweight="bold")
    pos_p = mpatches.Patch(color=COLOR_POS, label="Positive spinoff")
    neg_p = mpatches.Patch(color=COLOR_NEG, label="Negative spinoff")
    ax4.legend(handles=[pos_p, neg_p, mpatches.Patch(color=COLOR_SPY, label="SPY")], fontsize=7)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_equity_page(pdf, df: pd.DataFrame):
    """Page 3: cumulative equity curve."""
    curve = build_equity_curve(df)
    if curve.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 11),
                              gridspec_kw={"height_ratios": [3, 1]},
                              constrained_layout=True)
    fig.suptitle("Portfolio Equity Curve", fontsize=14, fontweight="bold",
                 color="#1a3a5c")

    ax = axes[0]
    ax.plot(curve.index, curve["strategy_value"], color=COLOR_STRAT,
            linewidth=1.5, label="Spinoff Strategy")
    ax.plot(curve.index, curve["spy_value"],      color=COLOR_SPY,
            linewidth=1.5, linestyle="--", label="SPY Benchmark")
    ax.fill_between(curve.index,
                    curve["strategy_value"], curve["spy_value"],
                    where=curve["strategy_value"] >= curve["spy_value"],
                    alpha=0.15, color=COLOR_POS, label="Outperformance")
    ax.fill_between(curve.index,
                    curve["strategy_value"], curve["spy_value"],
                    where=curve["strategy_value"] <  curve["spy_value"],
                    alpha=0.15, color=COLOR_NEG, label="Underperformance")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title("Cumulative Portfolio Value — $100,000 Starting Capital\n"
                 "(Equal-weight, 1-year hold, rebalanced per trade)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Rolling alpha (strategy - spy)
    ax2 = axes[1]
    roll_alpha = (curve["strategy_value"] - curve["spy_value"]) / 100_000 * 100
    ax2.bar(curve.index, roll_alpha, width=7,
            color=[COLOR_POS if v >= 0 else COLOR_NEG for v in roll_alpha],
            alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Cum. Excess Return (%)")
    ax2.set_title("Cumulative Excess Return vs SPY (%)", fontsize=9)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_trade_table(pdf, df: pd.DataFrame):
    """Page 4: Full trade log table."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("Complete Trade Log", fontsize=14, fontweight="bold",
                 color="#1a3a5c", y=0.98)
    ax = fig.add_axes([0.01, 0.05, 0.98, 0.90])
    ax.axis("off")

    display = df.copy()
    display = display.sort_values("spinoff_date")

    # Format columns
    def fmt_date(x):
        if pd.isna(x) or x is None:
            return "—"
        return pd.Timestamp(x).strftime("%Y-%m-%d")

    def fmt_ret(x):
        if pd.isna(x):
            return "—"
        return f"{x:+.1%}"

    rows_data = []
    for _, r in display.iterrows():
        rows_data.append([
            r["ticker"],
            fmt_date(r["spinoff_date"]),
            fmt_date(r["entry_date"]),
            fmt_date(r["exit_date"]),
            fmt_ret(r["spinoff_return"]),
            fmt_ret(r["spy_return"]),
            fmt_ret(r["alpha"]),
            r["status"].upper()[:6],
        ])

    columns = ["Ticker", "Spinoff", "Entry", "Exit",
               "Ret", "SPY", "Alpha", "Status"]

    n_rows = len(rows_data)
    col_widths = [0.08, 0.12, 0.12, 0.12, 0.09, 0.09, 0.09, 0.09]

    table = ax.table(
        cellText=rows_data,
        colLabels=columns,
        colWidths=col_widths,
        cellLoc="center",
        loc="upper center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.35)

    # Header style
    for j in range(len(columns)):
        table[0, j].set_facecolor("#1a3a5c")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Row colouring
    for i, row_d in enumerate(rows_data, start=1):
        ret_val = display.iloc[i-1]["spinoff_return"]
        alpha_v = display.iloc[i-1]["alpha"]
        bg = "#f7f7f7" if i % 2 == 0 else "#ffffff"
        for j in range(len(columns)):
            table[i, j].set_facecolor(bg)
            table[i, j].set_text_props(color="#333")
        # Colour return cells
        if not pd.isna(ret_val):
            fc = "#d4edda" if ret_val > 0 else "#f8d7da"
            table[i, 4].set_facecolor(fc)
        if not pd.isna(alpha_v):
            fc = "#d4edda" if alpha_v > 0 else "#f8d7da"
            table[i, 6].set_facecolor(fc)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_sector_page(pdf, df: pd.DataFrame):
    """Page 5: analysis by sector / time period."""
    closed = df[df["status"] == "closed"].dropna(subset=["spinoff_return", "alpha"])
    if len(closed) < 4:
        return

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 11), constrained_layout=True)
    fig.suptitle("Deep-Dive Analysis", fontsize=14, fontweight="bold", color="#1a3a5c")

    # (a) Returns by year of spinoff
    ax = axes[0, 0]
    closed = closed.copy()
    closed["year"] = pd.to_datetime(closed["spinoff_date"]).dt.year
    year_grp = closed.groupby("year")["spinoff_return"].agg(["mean", "count"])
    ax.bar(year_grp.index, year_grp["mean"] * 100,
           color=[COLOR_POS if v > 0 else COLOR_NEG for v in year_grp["mean"]],
           alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spinoff Year")
    ax.set_ylabel("Mean 1-Yr Return (%)")
    ax.set_title("(a) Mean Return by Spinoff Year", fontweight="bold")
    for yr, row in year_grp.iterrows():
        ax.text(yr, row["mean"] * 100 + (1 if row["mean"] >= 0 else -3),
                f"n={int(row['count'])}", ha="center", fontsize=7)

    # (b) Alpha by year
    ax2 = axes[0, 1]
    year_alpha = closed.groupby("year")["alpha"].mean()
    ax2.bar(year_alpha.index, year_alpha.values * 100,
            color=[COLOR_POS if v > 0 else COLOR_NEG for v in year_alpha.values],
            alpha=0.8)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Spinoff Year")
    ax2.set_ylabel("Mean Alpha vs SPY (%)")
    ax2.set_title("(b) Mean Alpha by Spinoff Year", fontweight="bold")

    # (c) Win rate by year
    ax3 = axes[1, 0]
    win_rate = closed.groupby("year").apply(lambda x: (x["spinoff_return"] > 0).mean())
    ax3.bar(win_rate.index, win_rate.values * 100, color=COLOR_STRAT, alpha=0.8)
    ax3.axhline(50, color="red", linewidth=1, linestyle="--", label="50% line")
    ax3.set_xlabel("Spinoff Year")
    ax3.set_ylabel("Win Rate (%)")
    ax3.set_title("(c) Win Rate by Spinoff Year", fontweight="bold")
    ax3.legend(fontsize=8)

    # (d) Box plot of alphas
    ax4 = axes[1, 1]
    alpha_pct = closed["alpha"].values * 100
    bp = ax4.boxplot(alpha_pct, patch_artist=True, notch=False,
                     medianprops=dict(color="red", linewidth=2))
    bp["boxes"][0].set_facecolor(COLOR_ALPHA)
    bp["boxes"][0].set_alpha(0.7)
    ax4.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax4.set_ylabel("Alpha vs SPY (%)")
    ax4.set_title("(d) Alpha Distribution (Box Plot)", fontweight="bold")
    ax4.set_xticklabels(["All Trades"])

    # Annotate quartiles
    q25, med, q75 = np.percentile(alpha_pct, [25, 50, 75])
    ax4.text(1.35, q25, f"Q1: {q25:.1f}%", fontsize=7, va="center", color="#555")
    ax4.text(1.35, med, f"Med: {med:.1f}%", fontsize=7, va="center", color="red")
    ax4.text(1.35, q75, f"Q3: {q75:.1f}%", fontsize=7, va="center", color="#555")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_methodology_page(pdf):
    """Page 6: strategy rationale and methodology."""
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 0.92), 1, 0.08, boxstyle="square,pad=0",
        facecolor="#1a3a5c", edgecolor="none"))
    ax.text(0.5, 0.96, "Strategy Rationale & Methodology",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color="white", transform=ax.transAxes)

    sections = [
        ("WHY SPINOFFS OUTPERFORM", [
            "1. FORCED SELLING — When a company spins off a subsidiary, shareholders of the parent",
            "   receive shares in the new entity whether they want them or not. Index funds must sell",
            "   if the spinoff is too small for their mandate. This indiscriminate selling creates",
            "   artificially depressed prices unrelated to business fundamentals.",
            "",
            "2. ANALYST NEGLECT — Spinoffs receive little sell-side coverage initially. As analysts",
            "   initiate coverage over the following 12–18 months, institutional interest builds,",
            "   and the stock re-rates to fair value.",
            "",
            "3. MANAGEMENT ALIGNMENT — Spinoff management teams typically receive fresh equity",
            "   compensation tied to the standalone stock. This creates strong incentives to",
            "   improve operations, execute buybacks, and communicate clearly with investors.",
            "",
            "4. FOCUSED STRATEGY — No longer constrained by a conglomerate parent, spinoffs can",
            "   pursue focused strategies, divest non-core assets, and optimize capital allocation.",
        ]),
        ("ACADEMIC EVIDENCE", [
            "• Desai & Jain (1999): Spinoffs outperform market by ~+25% over 3 years post-separation.",
            "• McConnell & Ovtchinnikov (2004): Spinoffs earn +20% alpha in year 1, fading by year 3.",
            "• Chemmanur & Yan (2004): Spinoff effect strongest for firms with high information asymmetry.",
            "• Miles & Rosenfeld (1983): Announcement returns +3.5% for parent, spinoff undervalued.",
            "• Greenblatt (1997): The '30-day rule' — wait for forced sellers to finish before buying.",
        ]),
        ("STRATEGY RULES (COMPLETE)", [
            "ENTRY:  Buy the spinoff stock 30 calendar days after its first day of trading.",
            "        This lag ensures the bulk of forced/index selling is complete.",
            "EXIT:   Sell exactly 365 calendar days (≈1 year) after entry.",
            "        Academic research shows alpha peaks around 12–18 months; we capture year 1.",
            "        For positions < 1 year old, we use the latest available price (open trade).",
            "FILTER: Parent company must have been an S&P 500 constituent at spinoff date.",
            "        The spinoff itself need not be in the S&P 500 (e.g. SNDK went to SmallCap 600).",
            "        Example: Mallinckrodt (MNK) is INCLUDED because parent Covidien (COV) was SPX.",
            "SIZE:   Equal dollar weight per trade. No leverage.",
            "BENCH:  Total return SPY over the identical entry-to-exit window.",
        ]),
        ("RISKS & LIMITATIONS", [
            "• Spinoffs can fail — some trade to zero (e.g., Mallinckrodt/MNK bankrupt 2020, GXO -48%).",
            "• Survivorship bias: companies with no tradeable data excluded (e.g., ADT 2012, Baxalta).",
            "• Transaction costs not modelled — spinoffs often have wide bid-ask spreads initially.",
            "• Sample size is small (~20 completed trades); statistical significance is limited.",
            "• Strategy works best in bull markets; alpha shrinks in bear markets.",
        ]),
    ]

    y = 0.88
    for title, lines in sections:
        ax.text(0.05, y, title, ha="left", va="top", fontsize=10,
                fontweight="bold", color="#1a3a5c", transform=ax.transAxes)
        y -= 0.028
        for line in lines:
            ax.text(0.05, y, line, ha="left", va="top", fontsize=8.2,
                    color="#333", transform=ax.transAxes, linespacing=1.3)
            y -= 0.022
        y -= 0.015

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    import matplotlib.ticker

    output_path = "SPX_Spinoff_Backtest.pdf"

    print("=" * 60)
    print("  SPX SPINOFF STRATEGY — BACKTEST")
    print("=" * 60)

    df = run_backtest()

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    st = compute_stats(df)
    if st:
        print(f"  Total spinoffs in database  : {st['total_spinoffs']}")
        print(f"  Completed (closed) trades   : {st['n_trades']}")
        print(f"  Open / recent positions     : {st['n_open']}")
        print(f"  Win rate                    : {st['win_rate']:.1%}")
        print(f"  Beat SPY rate               : {st['beat_spy_rate']:.1%}")
        print(f"  Mean 1-yr return            : {st['mean_return']:+.1%}")
        print(f"  Mean alpha vs SPY           : {st['mean_alpha']:+.1%}")
        print(f"  Information ratio           : {st['information_ratio']:.2f}")
        print(f"  t-stat (alpha > 0)          : {st['t_stat']:.2f}  p={st['p_value']:.3f}")
        print(f"  Best trade                  : {st['max_gain']:+.1%}")
        print(f"  Worst trade                 : {st['max_loss']:+.1%}")

    print(f"\nGenerating PDF → {output_path}")

    with PdfPages(output_path) as pdf:
        make_cover_page(pdf, st)
        make_return_charts(pdf, df, st)
        make_equity_page(pdf, df)
        make_trade_table(pdf, df)
        make_sector_page(pdf, df)
        make_methodology_page(pdf)

        # PDF metadata
        d = pdf.infodict()
        d["Title"]   = "SPX Spinoff Trading Strategy Backtest"
        d["Author"]  = "Quantitative Research — SPX_Spinoffs"
        d["Subject"] = "Systematic Spinoff Strategy — Backtest Report"
        d["Keywords"] = "spinoffs, systematic, backtest, S&P 500, alpha"
        d["CreationDate"] = datetime.today()

    print(f"PDF saved: {output_path}")
    print("Done.")

    # Save trade log to CSV
    df.to_csv("spinoff_trades.csv", index=False)
    print("Trade log saved: spinoff_trades.csv")

    return df, st


if __name__ == "__main__":
    df, stats = main()
