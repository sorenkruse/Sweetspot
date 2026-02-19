# 🌊 Volatility Scenario Analyzer

**Find the sweet spot for options during volatility spikes & crushes.**

Black-Scholes based option scenario analyzer with real-time market data from Yahoo Finance. Built for options traders who want to quickly evaluate which strike/DTE combination delivers the best risk/reward for a given market outlook.

## Features

### 🎯 Quick Forecast
- **Two sliders**: Set your price forecast (±30%) and VIX expectation
- **Auto VIX estimation**: Empirical SPX/VIX correlation with convexity for sell-offs
- **Sweet Spot cards**: Best ROI, best absolute P&L, best Vega/Theta ratio
- **Heatmap**: P&L % across Strike × DTE with adaptive color scaling
- **DTE curves**: Compare top strikes across expiration dates

### 🔥 Full Heatmap Analysis
- Configurable Strike × DTE grid
- Multiple metrics: P&L, P&L%, Vega, Delta
- Top 10 best combinations ranking

### 📋 Results Table
- Full results grid with all Greeks
- Sortable and filterable

### 📈 DTE Curves
- P&L%, Entry cost, and Vega exposure by DTE
- Multi-strike comparison

### 🎬 Multi-Day Simulation
- Simulate positions over multiple days
- Preset spot paths: Linear decline, sudden crash, V-recovery, slow grind
- Greeks evolution tracking

## Supported Positions
- **Long Put (LP)** — Crash protection / vol spike play
- **Long Call (LC)** — Bullish vol expansion
- **Short Put (SP)** — Income / vol crush play
- **Short Call (SC)** — Bearish income

## Installation

```bash
pip install -r requirements.txt
streamlit run vol_scenario_analyzer.py
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file: `vol_scenario_analyzer.py`
5. Deploy

## Requirements

- Python 3.9+
- See `requirements.txt`

## Tech Stack

- **Pricing**: Black-Scholes (no external pricing libs)
- **Market Data**: Yahoo Finance (yfinance)
- **UI**: Streamlit + Plotly
- **Math**: NumPy, SciPy

## Disclaimer

⚠️ For analysis purposes only — not financial advice. Options trading involves significant risk.

## License

MIT
