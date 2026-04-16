import os
import time
import random
import numpy as np
import yfinance as yf
from flask import Flask, render_template, request, jsonify
from openai import OpenAI, APIStatusError, RateLimitError

# Load .env file if present (OPENAI_API_KEY etc.)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)


# ── Candle classification ────────────────────────────────────────────────────

def classify_candle(row, prev_row=None):
    """Classify a single OHLC candle into a named candlestick pattern."""
    o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
    body = abs(c - o)
    total_range = h - l
    if total_range == 0:
        return "Doji"
    body_ratio = body / total_range
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    is_bullish = c >= o

    if body_ratio < 0.1:
        return "Doji"

    if body_ratio > 0.80:
        return "Bullish Marubozu" if is_bullish else "Bearish Marubozu"

    if prev_row is not None:
        po, pc = float(prev_row["Open"]), float(prev_row["Close"])
        prev_body = abs(pc - po)
        if is_bullish and pc < po and body > prev_body * 0.9 and c > po and o < pc:
            return "Bullish Engulfing"
        if not is_bullish and pc > po and body > prev_body * 0.9 and c < po and o > pc:
            return "Bearish Engulfing"

    if lower_wick > body * 2 and upper_wick < body * 0.5:
        return "Hammer" if is_bullish else "Inverted Hammer"
    if upper_wick > body * 2 and lower_wick < body * 0.5:
        return "Shooting Star" if not is_bullish else "Inverted Hammer"

    return "Bullish Candle" if is_bullish else "Bearish Candle"


# ── ATR ──────────────────────────────────────────────────────────────────────

def calculate_atr(df, period=14):
    """Compute Average True Range (ATR) as a rolling mean of true range values."""
    prev_close = df["Close"].shift(1)
    hl = df["High"] - df["Low"]
    hc = (df["High"] - prev_close).abs()
    lc = (df["Low"] - prev_close).abs()
    tr = np.maximum(hl, np.maximum(hc, lc))
    return tr.rolling(period).mean()


# ── VWAP ─────────────────────────────────────────────────────────────────────

def calculate_vwap(df):
    """Calculate running VWAP from high, low, close, and volume columns."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()


# ── Order-book proxy ─────────────────────────────────────────────────────────

def estimate_ob_imbalance(df):
    """Estimate recent buy/sell pressure from candle-direction volume over the last 10 bars."""
    recent = df.tail(10)
    buy_vol = float(recent[recent["Close"] > recent["Open"]]["Volume"].sum())
    sell_vol = float(recent[recent["Close"] <= recent["Open"]]["Volume"].sum())
    total = buy_vol + sell_vol
    if total == 0:
        return "Neutral (0%)"
    pct = (buy_vol - sell_vol) / total * 100
    if pct > 5:
        return f"+{pct:.0f}% Bid side"
    if pct < -5:
        return f"{abs(pct):.0f}% Ask side"
    return f"Neutral ({pct:+.0f}%)"


# ── Main data-fetch + context assembly ───────────────────────────────────────

def build_context(ticker: str) -> dict:
    """Fetch intraday market data and assemble model input context for a ticker."""
    stock = yf.Ticker(ticker)
    df = stock.history(period="5d", interval="5m")
    if df.empty:
        raise ValueError(f"No 5-minute data found for '{ticker}'. Check the ticker symbol.")

    df = df.dropna()

    # Last 3 candles — reset_index so positional iloc works cleanly
    last3 = df.tail(3).reset_index(drop=True)
    candles = []
    for i in range(len(last3)):
        row = last3.iloc[i]
        prev = last3.iloc[i - 1] if i > 0 else None
        candles.append(classify_candle(row, prev))
    candle_str = " → ".join(f"5m {c}" for c in candles)

    # ATR state
    atr = calculate_atr(df)
    cur_atr = float(atr.iloc[-1])
    avg_atr = float(atr.tail(20).mean())
    if cur_atr > avg_atr * 1.12:
        atr_state = f"Expanding ({cur_atr:.4f})"
    elif cur_atr < avg_atr * 0.88:
        atr_state = f"Contracting ({cur_atr:.4f})"
    else:
        atr_state = f"Stable ({cur_atr:.4f})"

    # VWAP — filter to today using .date comparison (works with tz-aware index)
    last_date = df.index[-1].date()
    today = df[[d.date() == last_date for d in df.index]]
    if today.empty:
        today = df
    vwap_series = calculate_vwap(today)
    vwap_val = float(vwap_series.iloc[-1])
    price = float(df["Close"].iloc[-1])
    vwap_diff = (price - vwap_val) / vwap_val * 100
    direction = "above" if vwap_diff >= 0 else "below"
    vwap_str = f"{abs(vwap_diff):.2f}% {direction} VWAP (${vwap_val:.2f})"

    ob_str = estimate_ob_imbalance(df)

    return {
        "ticker": ticker.upper(),
        "price": round(price, 2),           # plain Python float now
        "candles": candle_str,
        "ob": ob_str,
        "atr": atr_state,
        "vwap": vwap_str,
    }


# ── OpenAI call ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a Transformer-based predictive engine specializing in market micro-structure. "
    "You do not analyze charts; instead, you treat market data sequences as 'sentences' where "
    "the 'next word' is the trade execution outcome. Be concise, precise, and use "
    "sequence-prediction terminology in your reasoning."
)

class UpstreamRateLimitError(Exception):
    """Raised when OpenAI rate limiting persists after retries."""
    pass


def _read_int_env(name: str, default: int, minimum: int = 1) -> int:
    """Read an integer environment variable with safe default and minimum bounds."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(int(raw), minimum)
    except ValueError:
        return default


def _read_float_env(name: str, default: float, minimum: float = 0.0) -> float:
    """Read a float environment variable with safe default and minimum bounds."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(float(raw), minimum)
    except ValueError:
        return default


_LAST_OPENAI_CALL_TS = 0.0


def _enforce_request_cooldown():
    """Pause briefly between OpenAI calls to reduce bursty request rate."""
    global _LAST_OPENAI_CALL_TS
    cooldown = _read_float_env("OPENAI_REQUEST_COOLDOWN_SEC", 1.0, minimum=0.0)
    now = time.monotonic()
    sleep_for = cooldown - (now - _LAST_OPENAI_CALL_TS)
    if sleep_for > 0:
        time.sleep(sleep_for)
    _LAST_OPENAI_CALL_TS = time.monotonic()


def _retry_delay_seconds(err: Exception, attempt: int) -> float:
    """Return retry delay using exponential backoff, jitter, and retry-after if present."""
    # Exponential backoff plus jitter to reduce synchronized retries.
    jitter = random.uniform(0, _read_float_env("OPENAI_RATE_LIMIT_JITTER_SEC", 0.5, minimum=0.0))
    backoff = (2 ** attempt) + jitter

    response = getattr(err, "response", None)
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                backoff = max(backoff, float(retry_after))
            except ValueError:
                pass

    return backoff


def run_prediction(ctx: dict) -> str:
    """Send assembled context to OpenAI and return the model prediction markdown."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Add it to a .env file in the project folder or export it in your shell."
        )

    client = OpenAI(api_key=api_key)
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    max_tokens = _read_int_env("OPENAI_MAX_TOKENS", 450, minimum=128)
    retries = _read_int_env("OPENAI_RATE_LIMIT_RETRIES", 3, minimum=0)

    user_prompt = f"""Ticker: {ctx['ticker']} — Current Price: ${ctx['price']}

Input Sequence (The "Context Window"):

Last 3 Candles: {ctx['candles']}
Order Book Imbalance: {ctx['ob']}
Volatility (ATR): {ctx['atr']}
Current Price vs. VWAP: {ctx['vwap']}

The Task:
Based on the statistical probability inherent in this sequence, predict the "Next Token" in \
the trade execution chain. Choose from the following outcomes:
1. Immediate Fill (Bullish Momentum)
2. Slippage (High Volatility)
3. Order Rejection/Fade (Resistance)
4. Stagnation (Liquidity Gap)

Output Requirements:
- **Prediction:** The most likely "Next Token."
- **Probability Distribution:** Assign a percentage to all 4 outcomes (must total 100%).
- **The "Why":** Explain the logic using sequence-prediction terminology (e.g., attention \
weights, token probabilities, context window signals).

Format your response with clear markdown headers for each section."""

    for attempt in range(retries + 1):
        try:
            _enforce_request_cooldown()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except RateLimitError as err:
            if attempt >= retries:
                raise UpstreamRateLimitError(
                    "OpenAI rate limit reached. Try again in a few seconds, or lower "
                    "OPENAI_MAX_TOKENS in your .env."
                ) from err
            time.sleep(_retry_delay_seconds(err, attempt))
        except APIStatusError as err:
            if err.status_code == 429:
                if attempt >= retries:
                    raise UpstreamRateLimitError(
                        "OpenAI rate limit reached. Try again in a few seconds, or lower "
                        "OPENAI_MAX_TOKENS in your .env."
                    ) from err
                time.sleep(_retry_delay_seconds(err, attempt))
                continue
            raise

    raise UpstreamRateLimitError("OpenAI rate limit reached. Please retry shortly.")


# ── Flask routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main single-page UI."""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Validate ticker input, build market context, run prediction, and return JSON."""
    data = request.get_json(silent=True) or {}
    ticker = (data.get("ticker") or "").strip().upper()
    if not ticker:
        return jsonify({"error": "Ticker symbol is required."}), 400
    try:
        ctx = build_context(ticker)
        prediction = run_prediction(ctx)
        return jsonify({"context": ctx, "prediction": prediction})
    except UpstreamRateLimitError as e:
        return jsonify({"error": str(e)}), 429
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5050)
