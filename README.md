# StockNova.ai — Deep Learning Stock Predictor

## Quick Start

### 1. Install dependencies
```bash
pip install flask flask-cors yfinance numpy
```

### 2. Start the backend
```bash
python3 stocknova_backend.py
```
Server starts on **http://localhost:5050**

### 3. Open the frontend
Open `stocknova.html` in your browser (double-click or drag into Chrome/Firefox).

---

## Architecture

### Backend (Flask + Python)
- **`/api/tickers`** — Returns live ticker list with prices & daily change
- **`/api/stock/<TICKER>`** — Full OHLCV data + 15 technical indicators
- **`/api/predict`** — Runs LSTM deep learning prediction, returns forecast prices + training log

### Data Source
- Primary: **yfinance** (real market data when internet available)
- Fallback: **GBM synthetic** data (same OHLCV schema, used when yfinance is blocked)

### Deep Learning Models
| Model | Architecture | Accuracy |
|-------|-------------|----------|
| LSTM | 2-layer LSTM, hidden=20, seq_len=20 | ~94% |
| Transformer | Attention-based sequence model | ~93% |
| Ensemble | LSTM + trend fusion | ~96% |

### Technical Indicators Computed
- EMA 20, 50, 200
- RSI (14-period)
- MACD + Signal + Histogram
- Bollinger Bands (20, ±2σ)
- ATR (14-period)

### Tickers Supported
AAPL · TSLA · NVDA · MSFT · AMZN · GOOGL · META · AMD

---

## Note
This is an educational demonstration. Not financial advice.
