from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import requests
import os
import logging
import re
from typing import Tuple

# ---- Basic config ----
app = Flask(__name__)
CORS(app)  # allow cross-origin requests if needed

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_server")

# Environment-configured API keys (set these in PythonAnywhere "Web -> Environment variables")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY")  # simple auth for your clients

# Network timeouts (seconds) for external API calls
API_TIMEOUT = 12

# ---- Utility helpers ----
def require_api_key():
    """Simple header-based auth to prevent misuse. Set BACKEND_API_KEY in env."""
    if not BACKEND_API_KEY:
        return  # if not set, skip checking (development convenience)
    header_key = request.headers.get("X-API-KEY")
    if not header_key or header_key != BACKEND_API_KEY:
        abort(401, description="Missing or invalid X-API-KEY header")

def parse_signal_and_confidence(text: str) -> dict:
    """Return {'signal': 'buy'|'sell'|'hold', 'confidence': float, 'raw': text}"""
    if not text:
        return {"signal": "hold", "confidence": 0.5, "raw": text}

    txt = text.lower()
    # look for explicit tokens
    if re.search(r'\bbuy\b', txt):
        signal = "buy"
    elif re.search(r'\bsell\b', txt):
        signal = "sell"
    else:
        signal = "hold"

    # attempt to extract a numeric confidence like "confidence: 0.78" or "78%"
    conf = None
    m = re.search(r'([0-9]{1,3})\s*%', text)
    if m:
        try:
            conf = max(0.0, min(1.0, float(m.group(1)) / 100.0))
        except:
            conf = None
    if conf is None:
        m2 = re.search(r'confidence[:=]\s*([0-9]*\.?[0-9]+)', text, re.I)
        if m2:
            try:
                v = float(m2.group(1))
                conf = v if 0.0 <= v <= 1.0 else (max(0.0, min(1.0, v/100.0)))
            except:
                conf = None

    if conf is None:
        conf = 0.95 if signal in ("buy", "sell") else 0.5

    return {"signal": signal, "confidence": round(conf, 2), "raw": text.strip()}

# ---- OpenAI (via REST) ----
def call_openai(prompt: str) -> Tuple[bool, str]:
    """Return (ok, text). ok=False on error."""
    if not OPENAI_API_KEY:
        return False, "OpenAI API key not configured"
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.2,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=API_TIMEOUT)
        r.raise_for_status()
        j = r.json()
        # flexibly fetch content
        content = ""
        try:
            content = j["choices"][0]["message"]["content"]
        except Exception:
            # defensive fallback if shape differs
            content = j.get("choices", [{}])[0].get("text", "")
        return True, content
    except Exception as e:
        logger.warning("OpenAI call failed: %s", e)
        return False, f"OpenAI error: {e}"

# ---- Groq (fallback) ----
def call_groq(prompt: str) -> Tuple[bool, str]:
    if not GROQ_API_KEY:
        return False, "Groq API key not configured"
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.2,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=API_TIMEOUT)
        r.raise_for_status()
        j = r.json()
        content = j.get("choices", [{}])[0].get("message", {}).get("content", "")
        # fallback to older structure
        if not content:
            content = j.get("choices", [{}])[0].get("text", "")
        return True, content
    except Exception as e:
        logger.warning("Groq call failed: %s", e)
        return False, f"Groq error: {e}"

# ---- Predict endpoint ----
@app.route("/predict", methods=["GET", "POST"])
def predict():
    try:
        require_api_key()
    except Exception as e:
        return jsonify({"error": "unauthorized", "details": str(e)}), 401

    # Accept both GET and POST
    if request.method == "POST":
        try:
            payload = request.get_json(force=True)
        except Exception:
            return jsonify({"error": "Invalid JSON body"}), 400
    else:
        payload = request.args.to_dict()

    symbol = (payload.get("symbol") or "XAUUSD").upper()
    timeframe = (payload.get("timeframe") or "H1").upper()
    market_data = payload.get("market_data") or payload.get("data") or ""

    # Build a clear instruction: request simple output with signal + optional confidence
    prompt = (
        f"You are an expert market analyst. Analyze {symbol} on timeframe {timeframe}."
        f" Use the following market data (if provided): {market_data}."
        " Give a short output stating BUY, SELL, or HOLD, and include a confidence value between 0.0 and 1.0."
        " Example answer: 'BUY — confidence: 0.82' or 'HOLD — confidence: 0.50'."
        " Keep the answer short."
    )

    # Try providers in preferred order: OpenAI -> Groq
    # If OpenAI available, use it. If not, try Groq. If both fail, return error.
    ok, text = call_openai(prompt)
    used_provider = "openai"
    if not ok:
        logger.info("OpenAI failed, trying Groq: %s", text)
        ok2, text2 = call_groq(prompt)
        if ok2:
            used_provider = "groq"
            text = text2
        else:
            logger.error("Both AI providers failed. OpenAI: %s | Groq: %s", text, text2)
            return jsonify({
                "symbol": symbol,
                "timeframe": timeframe,
                "error": "no_ai_provider_available",
                "details": {"openai": text, "groq": text2}
            }), 502

    parsed = parse_signal_and_confidence(text)

    response = {
        "symbol": symbol,
        "timeframe": timeframe,
        "provider": used_provider,
        "raw": parsed["raw"],
        "signal": parsed["signal"],
        "confidence": parsed["confidence"]
    }

    logger.info("Predicted %s %s %s (conf=%s) by %s", symbol, timeframe, parsed["signal"], parsed["confidence"], used_provider)
    return jsonify(response), 200

# ---- Only run locally for debugging ----
if __name__ == "__main__":
    # Local debug mode; do NOT enable in production on PythonAnywhere
    app.run(host="0.0.0.0", port=5000, debug=True)

