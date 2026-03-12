"""
SPX Spinoffs Telegram Agent — Self-Scheduling Poller

Mirrors daily-digest-telegram-webhook exactly:
  • polls Telegram every POLL_INTERVAL_S seconds
  • passes every message to a Claude agent that has tools:
      - run_backtest  — invoke spx-spinoffs-backtest Lambda
      - search_web    — DuckDuckGo search for current information
      - open_url      — fetch and read any URL
      - reply         — send a text reply to the user
  • self-reinvokes to stay alive (same DEPLOY_ID / watchdog pattern)

DEPLOY_ID: generation tag stamped at deploy time — stale chains exit immediately.
"""

import html as _html
import io
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── CloudWatch handler (mirrors daily-digest-telegram-webhook) ─────────────────

import boto3 as _boto3
import time as _time


class _CWHandler(logging.Handler):
    _LOG_GROUP  = '/aws/lambda/spx-spinoffs-telegram-agent'
    _LOG_STREAM = 'telegram-poller'
    _seq_token  = None

    def __init__(self):
        super().__init__()
        self._cw = _boto3.client('logs', region_name='eu-north-1')
        self._ensure_stream()

    def _ensure_stream(self):
        try:
            self._cw.create_log_stream(
                logGroupName=self._LOG_GROUP,
                logStreamName=self._LOG_STREAM,
            )
        except Exception:
            pass

    def emit(self, record):
        msg = self.format(record)
        kwargs = dict(
            logGroupName=self._LOG_GROUP,
            logStreamName=self._LOG_STREAM,
            logEvents=[{'timestamp': int(_time.time() * 1000), 'message': msg}],
        )
        if self._seq_token:
            kwargs['sequenceToken'] = self._seq_token
        try:
            resp = self._cw.put_log_events(**kwargs)
            self.__class__._seq_token = resp.get('nextSequenceToken')
        except Exception:
            pass


try:
    _cw_handler = _CWHandler()
    _cw_handler.setFormatter(logging.Formatter('[SPX-AGENT] %(levelname)s %(message)s'))
    logger.addHandler(_cw_handler)
except Exception:
    pass


# ── Config ─────────────────────────────────────────────────────────────────────

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID   = os.environ.get('TELEGRAM_CHAT_ID', '')
ANTHROPIC_API_KEY  = os.environ.get('ANTHROPIC_API_KEY', '')
REGION             = os.environ.get('AWS_REGION', 'eu-north-1')
DEPLOY_ID          = os.environ.get('DEPLOY_ID', '')
BACKTEST_LAMBDA    = os.environ.get('BACKTEST_LAMBDA', 'spx-spinoffs-backtest')

POLL_INTERVAL_S    = 60
BUSY_WAIT_S        = 300
CYCLES_PER_RUN     = 12
MAX_AGE_SECONDS    = 180
REINVOKE_BUFFER_MS = 90_000
MAX_AGENT_TURNS    = 8   # max Claude ↔ tool round-trips per message

SYSTEM_PROMPT = """\
You are the SPX Spinoff Strategy Assistant — a financial research agent \
focused on the systematic spinoff trading strategy documented in the \
SPX_Spinoffs project.

You have access to the following tools:
  • run_backtest — triggers a fresh backtest run (downloads live data, \
    generates a PDF report, emails it, and stores it in S3)
  • search_web   — searches the internet for current news, filings, or data
  • open_url     — fetches and reads the content of a specific URL
  • reply        — sends your final answer to the user

Strategy context:
  Universe:  S&P 500 spinoffs (parent must be S&P 500 member at spinoff date)
  Entry:     Buy 30 calendar days after first trading day
  Exit:      Sell exactly 1 year after entry
  Benchmark: SPY (S&P 500 ETF)
  Rationale: Index-fund selling creates a predictable supply/demand \
    imbalance that reverses over the following 12 months.

Latest backtest results (closed trades):
  Win rate: 73.9% | Mean return: +28.8% | Mean alpha vs SPY: +12.4%
  Open positions: SNDK, SOLS, Q

When users ask about current prices, news, or recent spinoff events, \
use search_web to get up-to-date information before replying. \
Always use reply as your final action to send the answer to the user.\
"""


# ── Telegram helpers ───────────────────────────────────────────────────────────

def tg_post(method, payload):
    url  = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data,
                                  headers={'Content-Type': 'application/json'},
                                  method='POST')
    resp = urllib.request.urlopen(req, timeout=15)
    return json.loads(resp.read())


def send_message(chat_id, text, parse_mode='Markdown'):
    # Telegram limit is 4096 chars; chunk if needed
    for chunk in _chunk_text(text, 4000):
        try:
            tg_post('sendMessage', {
                'chat_id':    chat_id,
                'text':       chunk,
                'parse_mode': parse_mode,
            })
        except Exception as e:
            # Retry without parse_mode in case of formatting error
            try:
                tg_post('sendMessage', {'chat_id': chat_id, 'text': chunk})
            except Exception as e2:
                logger.error("send_message failed: %s / %s", e, e2)


def _chunk_text(text, size):
    """Split text into chunks of at most `size` characters."""
    for i in range(0, max(len(text), 1), size):
        yield text[i:i + size]


def get_fresh_offset():
    try:
        result = tg_post('getUpdates', {'limit': 1, 'timeout': 0})
        updates = result.get('result', [])
        if updates:
            return updates[-1]['update_id'] + 1
    except Exception as e:
        logger.warning("Could not fetch fresh offset: %s", e)
    return 0


# ── Anthropic helpers ──────────────────────────────────────────────────────────

def _anthropic_post(body, timeout=90):
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        'https://api.anthropic.com/v1/messages',
        data=data,
        headers={
            'x-api-key':         ANTHROPIC_API_KEY,
            'anthropic-version': '2023-06-01',
            'content-type':      'application/json',
        },
        method='POST',
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read())


# ── Internet tools ─────────────────────────────────────────────────────────────

def search_web(query: str, max_results: int = 6) -> str:
    """Search via DuckDuckGo HTML interface and return a text summary."""
    encoded = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded}&kl=us-en"
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0',
        'Accept':     'text/html,application/xhtml+xml',
        'Accept-Language': 'en-US,en;q=0.9',
    })
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            raw = resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        return f"[search_web error: {e}]"

    # Extract result titles + snippets from DuckDuckGo HTML
    results = []
    # Titles: <a class="result__a" href="...">TITLE</a>
    # Snippets: <a class="result__snippet" ...>SNIPPET</a>
    title_re   = re.compile(r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', re.DOTALL)
    snippet_re = re.compile(r'class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL)

    titles   = title_re.findall(raw)
    snippets = [m.group(1) for m in snippet_re.finditer(raw)]

    def clean(s):
        s = re.sub(r'<[^>]+>', '', s)
        return _html.unescape(s).strip()

    for i, ((href, title), snippet) in enumerate(zip(titles[:max_results], snippets[:max_results])):
        results.append(f"{i+1}. {clean(title)}\n   {clean(snippet)}\n   {href}")

    if not results:
        # Fall back to DuckDuckGo Instant Answer API
        try:
            ia_url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_redirect=1&no_html=1"
            with urllib.request.urlopen(ia_url, timeout=10) as r:
                ia = json.loads(r.read())
            abstract = ia.get('AbstractText', '') or ia.get('Answer', '')
            if abstract:
                return f"DuckDuckGo Instant Answer:\n{abstract}"
        except Exception:
            pass
        return "[No search results found]"

    return f"Search results for: {query}\n\n" + "\n\n".join(results)


def open_url(url: str, max_chars: int = 4000) -> str:
    """Fetch a URL and return stripped text content."""
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0',
        'Accept': 'text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8',
    })
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            raw = resp.read(200_000).decode('utf-8', errors='replace')
    except urllib.error.HTTPError as e:
        return f"[HTTP {e.code}: {e.reason}]"
    except Exception as e:
        return f"[open_url error: {e}]"

    # Strip scripts, styles, then all tags
    text = re.sub(r'<script[^>]*>.*?</script>', ' ', raw, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>',  ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = _html.unescape(text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars] + '\n…[truncated]'
    return text or '[empty page]'


# ── Tool definitions ───────────────────────────────────────────────────────────

TOOLS = [
    {
        'name': 'run_backtest',
        'description': (
            'Run the SPX Spinoff Strategy backtest. Downloads fresh market data, '
            'runs the full backtest, uploads a PDF report to S3, and emails it. '
            'Use when the user asks to run, refresh, or re-run the backtest.'
        ),
        'input_schema': {'type': 'object', 'properties': {}, 'required': []},
    },
    {
        'name': 'search_web',
        'description': (
            'Search the internet for current information — news, stock data, '
            'spinoff announcements, SEC filings, financial results, etc.'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'Specific, targeted search query.',
                },
            },
            'required': ['query'],
        },
    },
    {
        'name': 'open_url',
        'description': 'Fetch and read the text content of a specific URL.',
        'input_schema': {
            'type': 'object',
            'properties': {
                'url': {
                    'type': 'string',
                    'description': 'Full URL to fetch (https://...).',
                },
            },
            'required': ['url'],
        },
    },
    {
        'name': 'reply',
        'description': (
            'Send your final answer to the user. Always call this as the '
            'last action. Markdown formatting is supported.'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'text': {
                    'type': 'string',
                    'description': 'Reply text. Markdown OK. Max ~3900 chars.',
                },
            },
            'required': ['text'],
        },
    },
]


# ── Backtest invoker ───────────────────────────────────────────────────────────

def _invoke_backtest(chat_id, lam_client):
    try:
        send_message(chat_id,
            "⏳ *Running SPX Spinoff backtest…*\n"
            "This takes ~5–10 minutes. I'll send a confirmation when done.")
    except Exception as e:
        logger.error("Failed to send ack: %s", e)
    try:
        resp = lam_client.invoke(
            FunctionName=BACKTEST_LAMBDA,
            InvocationType='Event',
            Payload=b'{}',
        )
        logger.info("Invoked %s — status %s", BACKTEST_LAMBDA, resp.get('StatusCode'))
    except Exception as e:
        logger.error("Failed to invoke backtest Lambda: %s", e)
        try:
            send_message(chat_id, f"❌ Failed to trigger backtest: {e}")
        except Exception:
            pass


# ── Agent loop ─────────────────────────────────────────────────────────────────

def run_agent(user_text: str, chat_id: str, lam_client) -> bool:
    """
    Run a multi-turn Claude agent loop.
    Returns True if the backtest was triggered (so caller enters busy-wait).
    """
    messages = [{'role': 'user', 'content': user_text}]
    report_triggered = False

    for turn in range(MAX_AGENT_TURNS):
        logger.info("Agent turn %d/%d", turn + 1, MAX_AGENT_TURNS)
        try:
            result = _anthropic_post({
                'model':      'claude-haiku-4-5-20251001',
                'max_tokens': 1024,
                'system':     SYSTEM_PROMPT,
                'messages':   messages,
                'tools':      TOOLS,
                'tool_choice': {'type': 'auto'},
            })
        except Exception as e:
            logger.error("Anthropic API error on turn %d: %s", turn + 1, e)
            try:
                send_message(chat_id, f"❌ API error: {e}")
            except Exception:
                pass
            return report_triggered

        content = result.get('content', [])
        stop_reason = result.get('stop_reason', '')

        # ── end_turn: Claude finished with plain text ──────────────────────
        if stop_reason == 'end_turn':
            for block in content:
                if block.get('type') == 'text' and block.get('text', '').strip():
                    send_message(chat_id, block['text'])
            return report_triggered

        # ── tool_use: Claude wants to call tools ───────────────────────────
        tool_blocks = [b for b in content if b.get('type') == 'tool_use']
        if not tool_blocks:
            # Unexpected: no tools and not end_turn — send any text and exit
            for block in content:
                if block.get('type') == 'text' and block.get('text', '').strip():
                    send_message(chat_id, block['text'])
            return report_triggered

        # Append assistant's message to conversation
        messages.append({'role': 'assistant', 'content': content})

        # Execute each tool and collect results
        tool_results = []
        done = False
        for tb in tool_blocks:
            name     = tb['name']
            inp      = tb.get('input', {})
            tool_id  = tb['id']

            if name == 'reply':
                text = inp.get('text', '').strip()
                if text:
                    send_message(chat_id, text)
                tool_results.append({
                    'type': 'tool_result',
                    'tool_use_id': tool_id,
                    'content': 'Sent.',
                })
                done = True  # reply is always the final action

            elif name == 'run_backtest':
                _invoke_backtest(chat_id, lam_client)
                report_triggered = True
                tool_results.append({
                    'type': 'tool_result',
                    'tool_use_id': tool_id,
                    'content': 'Backtest triggered successfully.',
                })
                done = True

            elif name == 'search_web':
                query = inp.get('query', '')
                logger.info("search_web: %r", query)
                result_text = search_web(query)
                tool_results.append({
                    'type': 'tool_result',
                    'tool_use_id': tool_id,
                    'content': result_text,
                })

            elif name == 'open_url':
                url = inp.get('url', '')
                logger.info("open_url: %s", url)
                result_text = open_url(url)
                tool_results.append({
                    'type': 'tool_result',
                    'tool_use_id': tool_id,
                    'content': result_text,
                })

            else:
                tool_results.append({
                    'type': 'tool_result',
                    'tool_use_id': tool_id,
                    'content': f"Unknown tool: {name}",
                    'is_error': True,
                })

        # Add tool results to conversation
        messages.append({'role': 'user', 'content': tool_results})

        if done:
            return report_triggered

    # Exceeded max turns — send a fallback
    logger.warning("Agent exceeded MAX_AGENT_TURNS")
    try:
        send_message(chat_id, "⚠️ Reached maximum reasoning steps. Please try rephrasing.")
    except Exception:
        pass
    return report_triggered


# ── Polling loop ───────────────────────────────────────────────────────────────

def poll_once(lam_client, offset):
    """Poll Telegram once. Returns (next_offset, report_triggered)."""
    now = int(time.time())
    try:
        result = tg_post('getUpdates', {
            'offset':          offset,
            'limit':           10,
            'timeout':         0,
            'allowed_updates': ['message'],
        })
    except Exception as e:
        logger.error("getUpdates failed: %s", e)
        return offset, False

    updates = result.get('result', [])
    if not updates:
        return offset, False

    next_offset      = max(upd['update_id'] for upd in updates) + 1
    report_triggered = False

    for upd in updates:
        message = upd.get('message') or upd.get('edited_message')
        if not message:
            continue

        age = now - message.get('date', 0)
        if age > MAX_AGE_SECONDS:
            logger.info("Skipping stale message (age=%ds)", age)
            continue

        chat_id = str(message.get('chat', {}).get('id', ''))
        text    = (message.get('text') or '').strip()
        logger.info("Message chat_id=%s age=%ds: %r", chat_id, age, text)

        if chat_id != TELEGRAM_CHAT_ID:
            continue

        if not text:
            continue

        # /start and /help handled locally
        if text.startswith('/start') or text.startswith('/help'):
            try:
                send_message(chat_id,
                    '📊 *SPX Spinoffs Strategy Agent*\n\n'
                    'I can help you with the SPX Spinoff trading strategy. '
                    'Just write naturally:\n\n'
                    '• _"Run the backtest"_ — refresh the full backtest report\n'
                    '• _"What are the current open positions?"_\n'
                    '• _"Search for recent spinoff news"_\n'
                    '• _"What is the latest GEV stock price?"_\n'
                    '• _"Explain the strategy rationale"_\n'
                    '• `/help` — show this message')
            except Exception as e:
                logger.error("Failed to send help: %s", e)
            continue

        if not ANTHROPIC_API_KEY:
            send_message(chat_id, '❌ ANTHROPIC_API_KEY not configured.')
            continue

        if run_agent(text, chat_id, lam_client):
            report_triggered = True
            break  # one backtest per poll cycle

    return next_offset, report_triggered


# ── Lambda entrypoint (identical pattern to daily-digest-telegram-webhook) ─────

def lambda_handler(event, context):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Telegram env vars not set")
        return

    event_deploy_id = event.get('deploy_id')
    is_watchdog     = event_deploy_id is None
    if not is_watchdog and DEPLOY_ID and event_deploy_id != DEPLOY_ID:
        logger.info("Stale chain detected (event deploy_id=%r, current=%s) — stopping.",
                    event_deploy_id, DEPLOY_ID)
        return

    lam = boto3.client('lambda', region_name=REGION)

    offset = event.get('offset') or get_fresh_offset()

    logger.info("Starting loop: %d cycles × %ds  deploy_id=%s  offset=%d",
                CYCLES_PER_RUN, POLL_INTERVAL_S, DEPLOY_ID, offset)

    for cycle in range(CYCLES_PER_RUN):
        remaining_ms = context.get_remaining_time_in_millis()
        if remaining_ms < REINVOKE_BUFFER_MS:
            logger.info("Low time remaining (%dms) — breaking early to self-reinvoke.", remaining_ms)
            break

        logger.info("Cycle %d/%d (offset=%d)", cycle + 1, CYCLES_PER_RUN, offset)
        offset, report_triggered = poll_once(lam, offset)

        if report_triggered:
            slept = 0
            while slept < BUSY_WAIT_S:
                if context.get_remaining_time_in_millis() < REINVOKE_BUFFER_MS:
                    logger.info("Low time during busy wait — breaking early.")
                    break
                chunk = min(30, BUSY_WAIT_S - slept)
                time.sleep(chunk)
                slept += chunk
            logger.info("Busy mode over, resuming.")
        elif cycle < CYCLES_PER_RUN - 1:
            time.sleep(POLL_INTERVAL_S)

    if is_watchdog:
        logger.info("Watchdog run complete — exiting (EventBridge will fire next one).")
        return

    payload = {'offset': offset, 'deploy_id': DEPLOY_ID}
    logger.info("Re-invoking self (offset=%d, deploy_id=%s)…", offset, DEPLOY_ID)
    try:
        lam.invoke(
            FunctionName=context.function_name,
            InvocationType='Event',
            Payload=json.dumps(payload).encode(),
        )
        logger.info("Self-invocation scheduled")
    except Exception as e:
        logger.error("CRITICAL: self-invocation failed — polling will stop! %s", e)
        try:
            send_message(TELEGRAM_CHAT_ID,
                "⚠️ *Polling daemon stopped* — self-invocation failed.\n"
                "EventBridge watchdog will restart polling within 15 minutes.")
        except Exception:
            pass
