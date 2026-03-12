"""
AWS Lambda handler — SPX Spinoff Strategy Backtest
Runs the backtest weekly, generates a PDF report, and emails it via SES.

Environment variables (set in Lambda config):
  SENDER_EMAIL      - verified SES sender address
  RECIPIENT_EMAIL   - recipient address
  SES_REGION        - SES region (default: eu-west-1)
"""

import os
import sys
import logging
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

import boto3

# Lambda's /var/task is read-only; write all output to /tmp
os.chdir("/tmp")

log = logging.getLogger()
log.setLevel(logging.INFO)

SENDER    = os.environ.get("SENDER_EMAIL",    "marcin.zieba4@gmail.com")
RECIPIENT = os.environ.get("RECIPIENT_EMAIL", "marcin.zieba@yahoo.com")
REGION    = os.environ.get("SES_REGION",      "eu-west-1")

PDF_PATH = "/tmp/SPX_Spinoff_Backtest.pdf"


def handler(event, context):
    log.info("Starting SPX Spinoff backtest — %s", datetime.utcnow().isoformat())

    # Run the full backtest (writes PDF + CSV to /tmp/)
    from backtest import main as backtest_main
    backtest_main()

    if not os.path.exists(PDF_PATH):
        raise RuntimeError("Backtest did not produce PDF output at " + PDF_PATH)

    pdf_size = os.path.getsize(PDF_PATH)
    log.info("PDF generated: %d bytes — sending email to %s", pdf_size, RECIPIENT)

    with open(PDF_PATH, "rb") as f:
        pdf_bytes = f.read()

    today = datetime.utcnow().strftime("%Y-%m-%d")
    msg = MIMEMultipart("mixed")
    msg["Subject"] = f"SPX Spinoff Strategy — Backtest Report ({today})"
    msg["From"]    = SENDER
    msg["To"]      = RECIPIENT

    body_text = (
        f"Hi,\n\n"
        f"Please find attached the SPX Spinoff Strategy backtest report dated {today}.\n\n"
        f"Strategy summary:\n"
        f"  • Universe:  S&P 500 spinoffs (parent must be S&P 500 member at spinoff date)\n"
        f"  • Entry:     Buy 30 calendar days after first trading day\n"
        f"  • Exit:      Sell exactly 1 year after entry\n"
        f"  • Benchmark: SPY\n\n"
        f"This report was generated automatically by the SPX Spinoffs Lambda function.\n"
    )
    msg.attach(MIMEText(body_text, "plain"))

    attachment = MIMEBase("application", "pdf")
    attachment.set_payload(pdf_bytes)
    encoders.encode_base64(attachment)
    attachment.add_header(
        "Content-Disposition",
        "attachment",
        filename=f"SPX_Spinoff_Backtest_{today}.pdf",
    )
    msg.attach(attachment)

    ses = boto3.client("ses", region_name=REGION)
    ses.send_raw_email(
        Source=SENDER,
        Destinations=[RECIPIENT],
        RawMessage={"Data": msg.as_bytes()},
    )

    log.info("Email sent successfully.")
    return {"statusCode": 200, "body": f"Report emailed to {RECIPIENT}"}
