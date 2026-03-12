"""
AWS Lambda handler — SPX Spinoff Strategy Backtest
Runs the backtest, then:
  1. Uploads PDF report  → s3://s3bucketmz/Strategies/SPX_Spinoff_Backtest_<date>.pdf
  2. Writes JSON summary → s3://s3bucketmz/Strategies/json/spx_spinoffs_<date>.json
  3. Emails PDF to RECIPIENT_EMAIL via SES

Environment variables:
  SENDER_EMAIL      - verified SES sender address
  RECIPIENT_EMAIL   - recipient address
  SES_REGION        - SES region (default: eu-north-1)
  S3_BUCKET         - S3 bucket for output (default: s3bucketmz)
"""

import json
import math
import os
import logging
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timezone

import boto3
import numpy as np

# Lambda's /var/task is read-only; write all output to /tmp
os.chdir("/tmp")

log = logging.getLogger()
log.setLevel(logging.INFO)

SENDER    = os.environ.get("SENDER_EMAIL",    "marcin.zieba4@gmail.com")
RECIPIENT = os.environ.get("RECIPIENT_EMAIL", "marcin.zieba@yahoo.com")
REGION    = os.environ.get("SES_REGION",      "eu-north-1")
S3_BUCKET = os.environ.get("S3_BUCKET",       "s3bucketmz")

PDF_PATH  = "/tmp/SPX_Spinoff_Backtest.pdf"
JSON_PATH = "/tmp/spx_spinoffs.json"


def _safe(v):
    """Convert numpy/nan scalars to plain Python for JSON serialisation."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _build_json(df, stats: dict, today: str) -> dict:
    """
    Build a human-readable JSON summary of the strategy state.
    Includes:
      - run metadata
      - strategy description
      - aggregate performance statistics
      - current open positions (= active trades, the "current actions")
      - full closed trade log
      - excluded universe entries (sp500_parent=False)
    """
    from backtest import SPINOFFS  # filtered list (sp500_parent=True only)
    from backtest import _RAW_SPINOFFS  # full list including excluded

    # ── open / recent positions (the "current actions") ──────────────────
    open_df = df[df["status"] == "open"].copy()
    current_actions = []
    for _, row in open_df.iterrows():
        entry_dt = row["entry_date"]
        exit_dt  = row["exit_date"]
        current_ret = _safe(row.get("spinoff_return"))
        current_actions.append({
            "ticker":          row["ticker"],
            "parent":          row["parent"],
            "description":     row["description"],
            "spinoff_date":    str(row["spinoff_date"].date() if hasattr(row["spinoff_date"], "date") else row["spinoff_date"]),
            "entry_date":      str(entry_dt.date() if hasattr(entry_dt, "date") else entry_dt),
            "planned_exit":    str(exit_dt.date() if hasattr(exit_dt, "date") else exit_dt),
            "entry_price":     _safe(row.get("entry_price")),
            "current_return":  current_ret,
            "action":          "HOLD — exit at planned_exit date",
            "note":            (
                "Position is open. Return shown is mark-to-market vs. latest close. "
                "Strategy rule: sell exactly 365 days after entry."
            ),
        })

    # ── closed trade log ─────────────────────────────────────────────────
    closed_df = df[df["status"] == "closed"].copy()
    closed_trades = []
    for _, row in closed_df.iterrows():
        closed_trades.append({
            "ticker":         row["ticker"],
            "parent":         row["parent"],
            "description":    row["description"],
            "spinoff_date":   str(row["spinoff_date"].date() if hasattr(row["spinoff_date"], "date") else row["spinoff_date"]),
            "entry_date":     str(row["entry_date"].date() if hasattr(row["entry_date"], "date") else row["entry_date"]),
            "exit_date":      str(row["exit_date"].date() if hasattr(row["exit_date"], "date") else row["exit_date"]),
            "entry_price":    _safe(row.get("entry_price")),
            "exit_price":     _safe(row.get("exit_price")),
            "spinoff_return": _safe(row.get("spinoff_return")),
            "spy_return":     _safe(row.get("spy_return")),
            "alpha":          _safe(row.get("alpha")),
            "beat_spy":       bool((_safe(row.get("alpha")) or 0) > 0),
        })

    # ── excluded entries ──────────────────────────────────────────────────
    excluded = [
        {
            "ticker":  s["ticker"],
            "parent":  s["parent"],
            "date":    s["date"],
            "reason":  s.get("sp500_note", ""),
        }
        for s in _RAW_SPINOFFS if not s.get("sp500_parent", True)
    ]

    # ── stats: convert all values ─────────────────────────────────────────
    clean_stats = {k: _safe(v) for k, v in stats.items()}

    return {
        "metadata": {
            "generated_at":   datetime.now(timezone.utc).isoformat(),
            "report_date":    today,
            "source":         "SPX_Spinoffs Lambda (eu-north-1)",
            "s3_pdf":         f"s3://{S3_BUCKET}/Strategies/SPX_Spinoff_Backtest_{today}.pdf",
        },
        "strategy": {
            "name":      "SPX Spinoff Strategy",
            "universe":  "Spinoffs where parent was S&P 500 member at spinoff date",
            "entry":     "Buy 30 calendar days after first trading day",
            "exit":      "Sell exactly 365 calendar days after entry",
            "sizing":    "Equal weight; one position per symbol",
            "benchmark": "SPY (S&P 500 ETF)",
            "rationale": (
                "Spinoff shares are dumped by index funds & institutional holders "
                "who received shares they didn't ask for. This supply/demand "
                "imbalance reverses over the following 12 months as the business "
                "is re-discovered by the market."
            ),
            "references": [
                "Greenblatt (1997) — You Can Be a Stock Market Genius",
                "Desai & Jain (1999) — Journal of Financial Economics",
                "McConnell & Ovtchinnikov (2004) — JPM",
            ],
        },
        "performance": clean_stats,
        "current_actions": current_actions,
        "closed_trades":   closed_trades,
        "excluded_universe": excluded,
    }


def handler(event, context):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log.info("Starting SPX Spinoff backtest — %s", today)

    # ── 1. Run backtest ───────────────────────────────────────────────────
    from backtest import run_backtest, compute_stats
    df = run_backtest()
    stats = compute_stats(df)

    # Also trigger the PDF-generating main() separately
    from backtest import main as backtest_main
    backtest_main()

    if not os.path.exists(PDF_PATH):
        raise RuntimeError("Backtest did not produce PDF at " + PDF_PATH)

    log.info("PDF generated: %d bytes", os.path.getsize(PDF_PATH))

    # ── 2. Build JSON summary ─────────────────────────────────────────────
    payload = _build_json(df, stats, today)
    with open(JSON_PATH, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    log.info("JSON written: %d bytes", os.path.getsize(JSON_PATH))

    # ── 3. Upload to S3 ───────────────────────────────────────────────────
    s3 = boto3.client("s3")
    pdf_key  = f"Strategies/SPX_Spinoff_Backtest_{today}.pdf"
    json_key = f"Strategies/json/spx_spinoffs_{today}.json"

    s3.upload_file(PDF_PATH,  S3_BUCKET, pdf_key,
                   ExtraArgs={"ContentType": "application/pdf"})
    log.info("PDF uploaded  → s3://%s/%s", S3_BUCKET, pdf_key)

    s3.upload_file(JSON_PATH, S3_BUCKET, json_key,
                   ExtraArgs={"ContentType": "application/json"})
    log.info("JSON uploaded → s3://%s/%s", S3_BUCKET, json_key)

    # ── 4. Email PDF via SES ──────────────────────────────────────────────
    with open(PDF_PATH, "rb") as f:
        pdf_bytes = f.read()

    n_open   = len(df[df["status"] == "open"])
    n_closed = len(df[df["status"] == "closed"])
    mean_ret = stats.get("mean_return", float("nan"))
    mean_alpha = stats.get("mean_alpha", float("nan"))

    open_summary = ""
    open_df = df[df["status"] == "open"]
    if not open_df.empty:
        lines = []
        for _, row in open_df.iterrows():
            ret = row.get("spinoff_return")
            ret_str = f"{ret:+.1%}" if ret is not None and not (isinstance(ret, float) and math.isnan(ret)) else "n/a"
            lines.append(f"    {row['ticker']:6s}  entry {row['entry_date'].date()}  curr return {ret_str}")
        open_summary = "Current open positions (HOLD until exit date):\n" + "\n".join(lines) + "\n\n"

    body_text = (
        f"SPX Spinoff Strategy — Backtest Report {today}\n"
        f"{'=' * 55}\n\n"
        f"STRATEGY\n"
        f"  Universe:  S&P 500 spinoffs (parent in S&P 500 at spinoff date)\n"
        f"  Entry:     Buy 30 calendar days after first trading day\n"
        f"  Exit:      Sell exactly 1 year after entry\n"
        f"  Benchmark: SPY\n\n"
        f"PERFORMANCE SUMMARY ({n_closed} closed trades)\n"
        f"  Win rate:     {stats.get('win_rate', float('nan')):.1%}\n"
        f"  Beat SPY:     {stats.get('beat_spy_rate', float('nan')):.1%}\n"
        f"  Mean return:  {mean_ret:+.1%}\n"
        f"  Mean alpha:   {mean_alpha:+.1%}\n"
        f"  Info ratio:   {stats.get('information_ratio', float('nan')):.2f}\n"
        f"  t-stat:       {stats.get('t_stat', float('nan')):.2f}  "
        f"p={stats.get('p_value', float('nan')):.3f}\n\n"
        f"CURRENT ACTIONS ({n_open} open positions)\n"
        f"{open_summary}"
        f"S3 ARTIFACTS\n"
        f"  PDF:  s3://{S3_BUCKET}/{pdf_key}\n"
        f"  JSON: s3://{S3_BUCKET}/{json_key}\n\n"
        f"Full PDF report attached.\n"
    )

    msg = MIMEMultipart("mixed")
    msg["Subject"] = f"SPX Spinoff Strategy — Report {today}"
    msg["From"]    = SENDER
    msg["To"]      = RECIPIENT
    msg.attach(MIMEText(body_text, "plain"))

    attachment = MIMEBase("application", "pdf")
    attachment.set_payload(pdf_bytes)
    encoders.encode_base64(attachment)
    attachment.add_header("Content-Disposition", "attachment",
                          filename=f"SPX_Spinoff_Backtest_{today}.pdf")
    msg.attach(attachment)

    ses = boto3.client("ses", region_name=REGION)
    ses.send_raw_email(
        Source=SENDER,
        Destinations=[RECIPIENT],
        RawMessage={"Data": msg.as_bytes()},
    )
    log.info("Email sent to %s", RECIPIENT)

    return {
        "statusCode": 200,
        "pdf_s3":     f"s3://{S3_BUCKET}/{pdf_key}",
        "json_s3":    f"s3://{S3_BUCKET}/{json_key}",
        "open_positions": n_open,
        "closed_trades":  n_closed,
    }
