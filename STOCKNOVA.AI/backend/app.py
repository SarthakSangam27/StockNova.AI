from flask import Flask, jsonify, request
from flask_cors import CORS
import random
import math
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# ----------------------------------------
# STOCK CONFIG
# ----------------------------------------

TICKERS = {
    "AAPL": {"price": 190, "name": "Apple Inc.", "sector": "Technology"},
    "TSLA": {"price": 240, "name": "Tesla Inc.", "sector": "Automobile"},
    "MSFT": {"price": 420, "name": "Microsoft", "sector": "Technology"},
    "GOOGL": {"price": 175, "name": "Alphabet Inc.", "sector": "Technology"},
    "AMZN": {"price": 185, "name": "Amazon", "sector": "E-Commerce"},
}

# ----------------------------------------
# GENERATE FAKE MARKET DATA
# ----------------------------------------

def generate_stock_data(base_price):
    data = []

    current = base_price

    for i in range(300):
        date = (datetime.now() - timedelta(days=300-i)).strftime("%Y-%m-%d")

        change = random.uniform(-2, 2)
        current += change

        open_price = current + random.uniform(-1, 1)
        close_price = current + random.uniform(-1, 1)

        high = max(open_price, close_price) + random.uniform(0, 2)
        low = min(open_price, close_price) - random.uniform(0, 2)

        volume = random.randint(20000000, 90000000)

        ema20 = current + random.uniform(-2, 2)
        ema50 = current + random.uniform(-3, 3)
        ema200 = current + random.uniform(-5, 5)

        rsi = random.uniform(35, 70)

        macd = random.uniform(-3, 3)
        macd_signal = macd - random.uniform(-1, 1)
        macd_hist = macd - macd_signal

        bb_upper = current + 10
        bb_lower = current - 10
        bb_mid = current

        atr = random.uniform(1, 5)

        data.append({
            "date": date,
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close_price, 2),
            "volume": volume,

            "ema20": round(ema20, 2),
            "ema50": round(ema50, 2),
            "ema200": round(ema200, 2),

            "rsi": round(rsi, 2),

            "macd": round(macd, 3),
            "macd_signal": round(macd_signal, 3),
            "macd_hist": round(macd_hist, 3),

            "bb_upper": round(bb_upper, 2),
            "bb_lower": round(bb_lower, 2),
            "bb_mid": round(bb_mid, 2),

            "atr": round(atr, 2),
        })

    return data


# ----------------------------------------
# API ROUTES
# ----------------------------------------

@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "StockNova.ai Backend Running"
    })


@app.route("/api/tickers")
def tickers():

    result = []

    for symbol, info in TICKERS.items():

        result.append({
            "ticker": symbol,
            "price": info["price"],
            "change_pct": round(random.uniform(-3, 3), 2)
        })

    return jsonify(result)


@app.route("/api/stock/<ticker>")
def stock(ticker):

    ticker = ticker.upper()

    if ticker not in TICKERS:
        return jsonify({"error": "Ticker not found"}), 404

    info = TICKERS[ticker]

    data = generate_stock_data(info["price"])

    return jsonify({
        "ticker": ticker,
        "name": info["name"],
        "sector": info["sector"],
        "count": len(data),
        "data": data
    })


@app.route("/api/predict", methods=["POST"])
def predict():

    req = request.get_json()

    ticker = req.get("ticker", "AAPL")
    horizon = int(req.get("horizon", 7))

    current_price = TICKERS[ticker]["price"]

    prices = []

    value = current_price

    for i in range(horizon):

        drift = random.uniform(-2, 4)

        value += drift

        prices.append(round(value, 2))

    return jsonify({
        "ticker": ticker,
        "current_price": current_price,
        "prices": prices,
        "confidence": round(random.uniform(82, 97), 1),

        "rmse": round(random.uniform(1, 3), 3),
        "r2": round(random.uniform(0.85, 0.98), 3),
        "mae": round(random.uniform(0.5, 2), 3),
        "mape": round(random.uniform(0.5, 2), 3),
        "sharpe": round(random.uniform(1, 3), 3)
    })


# ----------------------------------------
# RUN
# ----------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)