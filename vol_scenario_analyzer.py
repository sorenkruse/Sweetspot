"""
Volatility Scenario Analyzer for Options
=========================================
Finds the sweet spot for long options during volatility spikes/crushes.
Uses Black-Scholes pricing with real market data from Yahoo Finance.
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Black-Scholes Engine
# ─────────────────────────────────────────────

def bs_d1(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def bs_d2(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    return bs_d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def bs_price(S, K, T, r, sigma, option_type="put"):
    """Black-Scholes option price."""
    if T <= 1e-10:
        if option_type == "call":
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_greeks(S, K, T, r, sigma, option_type="put"):
    """Calculate all Greeks."""
    if T <= 1e-10:
        intrinsic = max(K - S, 0) if option_type == "put" else max(S - K, 0)
        return {"price": intrinsic, "delta": -1.0 if option_type == "put" and S < K else 0.0,
                "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
    
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = bs_d2(S, K, T, r, sigma)
    sqrt_T = np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    
    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T / 100  # per 1% vol change
    
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-S * pdf_d1 * sigma / (2 * sqrt_T) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = (-S * pdf_d1 * sigma / (2 * sqrt_T) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    return {"price": price, "delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}


def implied_vol(price, S, K, T, r, option_type="put"):
    """Calculate implied volatility from option price."""
    if T <= 1e-10:
        return 0.0
    try:
        func = lambda sigma: bs_price(S, K, T, r, sigma, option_type) - price
        return brentq(func, 0.001, 5.0, xtol=1e-6)
    except:
        return None


# ─────────────────────────────────────────────
# Market Data
# ─────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_market_data(ticker_symbol):
    """Fetch current price and historical volatility from Yahoo Finance."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Current price
        hist = ticker.history(period="1y")
        if hist.empty:
            return None, None, None
        
        current_price = hist['Close'].iloc[-1]
        
        # Historical volatility (annualized, 30-day rolling)
        log_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        hv_30 = log_returns[-30:].std() * np.sqrt(252) if len(log_returns) >= 30 else log_returns.std() * np.sqrt(252)
        hv_60 = log_returns[-60:].std() * np.sqrt(252) if len(log_returns) >= 60 else hv_30
        
        return current_price, hv_30, hv_60
    except Exception as e:
        st.error(f"Error fetching data for {ticker_symbol}: {e}")
        return None, None, None


@st.cache_data(ttl=300)
def get_vix_level():
    """Fetch current VIX level."""
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="5d")
        if not hist.empty:
            return hist['Close'].iloc[-1]
    except:
        pass
    return 20.0  # default


def estimate_vix_from_spot_change(spot_change_pct, current_vix):
    """
    Estimate VIX level after a spot move based on empirical SPX/VIX relationship.
    Rule of thumb: VIX moves ~-4 to -5 points per +1% SPX (inverse, convex).
    For large drops the relationship is convex — VIX spikes disproportionately.
    """
    # Empirical approximation (based on historical regression):
    # VIX_change ≈ -4.5 * spot_change  for small moves
    # with convexity: for large negative moves, VIX spikes harder
    if spot_change_pct < 0:
        # Convex spike: quadratic boost for sell-offs
        linear = -4.0 * spot_change_pct
        convex_boost = 0.15 * spot_change_pct**2
        vix_change = linear + convex_boost
    else:
        # Rallies compress vol, but with a floor
        linear = -3.0 * spot_change_pct
        vix_change = linear
    
    new_vix = max(current_vix + vix_change, 9.0)  # VIX floor ~9
    return round(new_vix, 1)


def quick_forecast_analysis(S, r, current_vix, spot_change_pct, new_vix,
                             option_type="put", position="long",
                             min_dte=7, max_dte=365):
    """
    Simplified analysis: for a given forecast, find the best strike/DTE combo.
    Tests a focused grid and returns ranked results.
    """
    current_iv = current_vix / 100
    new_iv = new_vix / 100
    new_S = S * (1 + spot_change_pct / 100)
    
    # Smart strike range: centered around likely sweet spot
    if option_type == "put":
        # For puts: OTM to slightly ITM
        strike_center = S * (1 + spot_change_pct / 200)  # halfway to target
        strike_lo = S * 0.82
        strike_hi = S * 1.02
    else:
        strike_center = S * (1 + spot_change_pct / 200)
        strike_lo = S * 0.98
        strike_hi = S * 1.18
    
    step = max(10, int(round(S * 0.005 / 10) * 10))  # ~0.5% steps, rounded to 10
    if step < 10:
        step = 10
    strike_lo = int(np.floor(strike_lo / step) * step)
    strike_hi = int(np.ceil(strike_hi / step) * step)
    K_range = np.arange(strike_lo, strike_hi + step, step)
    
    # Typical DTE values, filtered by min/max
    all_dtes = [7, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365, 500, 730]
    dte_values = [d for d in all_dtes if min_dte <= d <= max_dte]
    if not dte_values:
        dte_values = [min_dte]
    
    results = []
    for K in K_range:
        for dte in dte_values:
            T_entry = dte / 365
            T_exit = max((dte - 1) / 365, 1/365)
            
            entry_price = bs_price(S, K, T_entry, r, current_iv, option_type)
            exit_price = bs_price(new_S, K, T_exit, r, new_iv, option_type)
            
            if position == "long":
                pnl = exit_price - entry_price
            else:
                pnl = entry_price - exit_price
            
            pnl_pct = (pnl / entry_price * 100) if entry_price > 0.001 else 0
            
            greeks = bs_greeks(S, K, T_entry, r, current_iv, option_type)
            
            moneyness = ((K / S) - 1) * 100
            
            # Leverage ratio: PnL per $100 invested
            leverage = pnl_pct / 100 if entry_price > 0.001 else 0
            
            results.append({
                "Strike": int(K),
                "DTE": dte,
                "Moneyness": f"{moneyness:+.1f}%",
                "Entry $": round(entry_price, 2),
                "Exit $": round(exit_price, 2),
                "P&L $": round(pnl, 2),
                "P&L %": round(pnl_pct, 1),
                "Delta": round(greeks["delta"], 3),
                "Vega": round(greeks["vega"], 2),
                "Theta/day": round(greeks["theta"], 2),
                "Vega/Theta": round(abs(greeks["vega"] / greeks["theta"]), 1) if abs(greeks["theta"]) > 0.001 else 0,
            })
    
    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# Scenario Analysis Engine
# ─────────────────────────────────────────────

def analyze_scenario(S, K_range, dte_range, r, current_iv, new_iv, spot_change_pct, 
                     option_type="put", position="long"):
    """
    Analyze P&L across strike/DTE grid for a volatility scenario.
    
    Returns DataFrame with columns: Strike, DTE, Entry_Price, Exit_Price, PnL, PnL_Pct, Greeks...
    """
    results = []
    new_S = S * (1 + spot_change_pct / 100)
    
    for K in K_range:
        for dte in dte_range:
            T_entry = dte / 365
            T_exit = max((dte - 1) / 365, 1/365)  # 1 day later for comparison
            
            # Entry at current IV
            entry_price = bs_price(S, K, T_entry, r, current_iv, option_type)
            
            # Exit at new IV and new spot
            exit_price = bs_price(new_S, K, T_exit, r, new_iv, option_type)
            
            if position == "long":
                pnl = exit_price - entry_price
            else:  # short
                pnl = entry_price - exit_price
            
            pnl_pct = (pnl / entry_price * 100) if entry_price > 0.01 else 0
            
            # Greeks at entry
            greeks = bs_greeks(S, K, T_entry, r, current_iv, option_type)
            
            # Moneyness
            if option_type in ["put"]:
                moneyness = (K / S - 1) * 100  # negative = OTM for puts
            else:
                moneyness = (S / K - 1) * 100  # negative = OTM for calls
            
            results.append({
                "Strike": K,
                "DTE": dte,
                "Moneyness_%": round(moneyness, 1),
                "Entry_Price": round(entry_price, 2),
                "Exit_Price": round(exit_price, 2),
                "PnL": round(pnl, 2),
                "PnL_%": round(pnl_pct, 1),
                "Delta": round(greeks["delta"], 4),
                "Gamma": round(greeks["gamma"], 6),
                "Vega": round(greeks["vega"], 2),
                "Theta": round(greeks["theta"], 2),
            })
    
    return pd.DataFrame(results)


def multi_day_scenario(S, K, dte, r, current_iv, vol_spike, spot_path_pct, 
                       option_type="put", position="long", days_forward=None):
    """
    Simulate option P&L over multiple days with gradual vol change and spot movement.
    spot_path_pct: array of cumulative % spot changes per day
    """
    if days_forward is None:
        days_forward = len(spot_path_pct)
    
    T_entry = dte / 365
    entry_price = bs_price(S, K, T_entry, r, current_iv, option_type)
    
    daily_results = []
    vol_increment = vol_spike / days_forward if days_forward > 0 else 0
    
    for day in range(days_forward + 1):
        T_now = max((dte - day) / 365, 1/365)
        current_vol = current_iv + vol_increment * day
        current_spot = S * (1 + spot_path_pct[min(day, len(spot_path_pct)-1)] / 100) if day < len(spot_path_pct) else S * (1 + spot_path_pct[-1] / 100)
        
        price_now = bs_price(current_spot, K, T_now, r, current_vol, option_type)
        greeks = bs_greeks(current_spot, K, T_now, r, current_vol, option_type)
        
        if position == "long":
            pnl = price_now - entry_price
        else:
            pnl = entry_price - price_now
        
        daily_results.append({
            "Day": day,
            "Spot": round(current_spot, 2),
            "IV": round(current_vol * 100, 1),
            "Price": round(price_now, 2),
            "PnL": round(pnl, 2),
            "PnL_%": round(pnl / entry_price * 100, 1) if entry_price > 0.01 else 0,
            "Delta": round(greeks["delta"], 4),
            "Vega": round(greeks["vega"], 2),
            "Theta": round(greeks["theta"], 2),
        })
    
    return pd.DataFrame(daily_results)


# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Vol Scenario Analyzer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=DM+Sans:wght@400;500;700&display=swap');
    
    .stApp {
        font-family: 'DM Sans', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: -0.5px;
    }
    
    .main-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1b5e20, #2e7d32, #4caf50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    
    .subtitle {
        font-family: 'DM Sans', sans-serif;
        color: #666;
        font-size: 0.95rem;
        margin-top: -8px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
    
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-title">🌊 Volatility Scenario Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Find the sweet spot for options during vol spikes & crushes — Black-Scholes based, real market data</p>', unsafe_allow_html=True)
st.markdown("---")

# ─── Sidebar ──────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Underlying
    ticker_input = st.text_input("Underlying", value="^SPX", help="Yahoo Finance ticker. Use ^SPX, ^NDX, AAPL, etc.")
    
    # Fetch market data
    spot_price, hv30, hv60 = get_market_data(ticker_input)
    vix_level = get_vix_level()
    
    if spot_price is None:
        # Fallback: try alternate tickers
        for alt in ["^GSPC", "SPY"]:
            spot_price, hv30, hv60 = get_market_data(alt)
            if spot_price is not None:
                st.info(f"Fallback: using {alt}")
                break
    
    if spot_price is None:
        st.error("Could not fetch market data. Using defaults.")
        spot_price = 6000.0
        hv30 = 0.18
        hv60 = 0.17
    
    st.markdown(f"**Spot:** {spot_price:,.2f}")
    st.markdown(f"**HV30:** {hv30*100:.1f}% · **HV60:** {hv60*100:.1f}%")
    st.markdown(f"**VIX:** {vix_level:.1f}")
    
    st.markdown("---")
    
    # Scenario Type
    st.markdown("### 🎯 Scenario")
    
    OPTION_LABELS = {
        "Long Put (LP)": ("put", "long"),
        "Long Call (LC)": ("call", "long"),
        "Short Put (SP)": ("put", "short"),
        "Short Call (SC)": ("call", "short"),
    }
    
    option_choice = st.selectbox("Option Type / Position", list(OPTION_LABELS.keys()), index=0)
    opt_type, opt_position = OPTION_LABELS[option_choice]
    
    # Preset scenarios
    PRESETS = {
        "Custom": {},
        "🔴 Crash (-10%, VIX→45)": {"spot_change": -10.0, "new_vol": 45.0},
        "🔴 Sharp Sell-off (-5%, VIX→35)": {"spot_change": -5.0, "new_vol": 35.0},
        "🟡 Mild Correction (-3%, VIX→25)": {"spot_change": -3.0, "new_vol": 25.0},
        "🟢 Vol Crush (+2%, VIX→14)": {"spot_change": 2.0, "new_vol": 14.0},
        "🟢 Rally (+5%, VIX→12)": {"spot_change": 5.0, "new_vol": 12.0},
        "⚪ Vol Spike (flat, VIX→40)": {"spot_change": 0.0, "new_vol": 40.0},
    }
    
    preset = st.selectbox("Preset Scenarios", list(PRESETS.keys()), index=0)
    
    st.markdown("---")
    
    # Manual scenario parameters
    st.markdown("### 📊 Scenario Parameters")
    
    default_spot_chg = PRESETS.get(preset, {}).get("spot_change", -5.0)
    default_new_vol = PRESETS.get(preset, {}).get("new_vol", 35.0)
    
    current_iv_input = st.slider("Current IV (%)", 5.0, 80.0, round(vix_level, 1), 0.5,
                                  help="Starting implied volatility")
    
    new_iv_input = st.slider("New IV after scenario (%)", 5.0, 120.0, default_new_vol, 0.5,
                              help="Target IV after the scenario plays out")
    
    spot_change = st.slider("Spot Change (%)", -30.0, 30.0, default_spot_chg, 0.5,
                             help="Percentage change in underlying price")
    
    risk_free = st.slider("Risk-Free Rate (%)", 0.0, 8.0, 4.5, 0.1) / 100
    
    st.markdown("---")
    
    # Strike/DTE ranges
    st.markdown("### 📐 Analysis Grid")
    
    # Strike range as % of spot
    strike_min_pct = st.slider("Strike range (% of Spot)", -30, 10, (-15, 5), 1,
                                help="Min/Max strike as % distance from spot")
    strike_step = st.selectbox("Strike step", [10, 25, 50, 100], index=2)
    
    # DTE range
    dte_min, dte_max = st.slider("DTE range", 1, 365, (7, 180), 1)
    dte_step = st.selectbox("DTE step", [1, 5, 7, 14, 30], index=2)


# ─── Main Analysis ──────────────────────────────────

# Build strike and DTE arrays
S = spot_price
strike_lo = S * (1 + strike_min_pct[0] / 100)
strike_hi = S * (1 + strike_min_pct[1] / 100)
# Round to step
strike_lo = int(np.floor(strike_lo / strike_step) * strike_step)
strike_hi = int(np.ceil(strike_hi / strike_step) * strike_step)
K_range = np.arange(strike_lo, strike_hi + strike_step, strike_step)

dte_range = np.arange(dte_min, dte_max + dte_step, dte_step)

current_iv = current_iv_input / 100
new_iv = new_iv_input / 100

# Scenario summary
new_spot = S * (1 + spot_change / 100)

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("Spot Now → After", f"{S:,.0f}", f"{spot_change:+.1f}% → {new_spot:,.0f}")
with col_m2:
    vol_chg = new_iv_input - current_iv_input
    st.metric("IV Now → After", f"{current_iv_input:.1f}%", f"{vol_chg:+.1f}% → {new_iv_input:.1f}%")
with col_m3:
    st.metric("Position", f"{option_choice}")
with col_m4:
    st.metric("Grid", f"{len(K_range)}×{len(dte_range)}", f"{len(K_range)*len(dte_range)} combos")

# Run scenario
df = analyze_scenario(S, K_range, dte_range, risk_free, current_iv, new_iv, 
                      spot_change, opt_type, opt_position)

# ─── Tabs ──────────────────────────────────
tab_forecast, tab_heat, tab_table, tab_curves, tab_sim = st.tabs([
    "🎯 Quick Forecast", "🔥 Heatmap", "📋 Results Table", "📈 DTE Curves", "🎬 Multi-Day Simulation"
])

# ─── TAB 0: QUICK FORECAST ──────────────────────────────────
with tab_forecast:
    st.markdown("### 🎯 Quick Forecast")
    
    # ── Forecast sliders — compact row ──
    fc_col1, fc_col2 = st.columns(2)
    
    with fc_col1:
        fc_spot_change = st.slider(
            "📉 Erwartete Spot-Veränderung (%)",
            min_value=-30.0, max_value=30.0, value=-5.0, step=0.5,
            key="fc_spot",
            help="Positive = Rally, Negative = Sell-off"
        )
    
    with fc_col2:
        fc_auto_vix = st.checkbox("VIX automatisch schätzen", value=True, key="fc_auto_vix",
                                   help="~4pt VIX pro 1% SPX, konvex bei Sell-offs")
        
        # Auto-estimate VIX from spot change
        estimated_vix = estimate_vix_from_spot_change(fc_spot_change, vix_level)
        
        if fc_auto_vix:
            fc_new_vix = estimated_vix
        else:
            fc_new_vix = st.slider(
                "🌊 VIX nach Szenario",
                min_value=9.0, max_value=90.0, value=estimated_vix, step=0.5,
                key="fc_vix_manual"
            )
    
    # ── Summary boxes — light, same height ──
    fc_new_spot = S * (1 + fc_spot_change / 100)
    vix_change = fc_new_vix - vix_level
    spot_color = "#16a34a" if fc_spot_change >= 0 else "#dc2626"
    vix_color = "#dc2626" if vix_change > 0 else "#16a34a"
    
    box_style = (
        "background:#f8f9fa; border:1px solid #dee2e6; "
        "border-radius:8px; padding:10px 16px; height:70px; display:flex; "
        "flex-direction:column; justify-content:center;"
    )
    
    sb1, sb2, sb3, sb4 = st.columns(4)
    with sb1:
        st.markdown(
            f'<div style="{box_style}">'
            f'<span style="color:#6c757d; font-size:0.7rem; font-family:JetBrains Mono;">SPOT</span>'
            f'<span style="font-family:JetBrains Mono; font-size:1.1rem; color:{spot_color};">'
            f'{S:,.0f} → {fc_new_spot:,.0f}</span></div>', unsafe_allow_html=True)
    with sb2:
        st.markdown(
            f'<div style="{box_style}">'
            f'<span style="color:#6c757d; font-size:0.7rem; font-family:JetBrains Mono;">SPOT Δ</span>'
            f'<span style="font-family:JetBrains Mono; font-size:1.1rem; color:{spot_color};">'
            f'{fc_spot_change:+.1f}% ({fc_new_spot - S:+,.0f})</span></div>', unsafe_allow_html=True)
    with sb3:
        st.markdown(
            f'<div style="{box_style}">'
            f'<span style="color:#6c757d; font-size:0.7rem; font-family:JetBrains Mono;">VIX</span>'
            f'<span style="font-family:JetBrains Mono; font-size:1.1rem; color:{vix_color};">'
            f'{vix_level:.1f} → {fc_new_vix:.1f}</span></div>', unsafe_allow_html=True)
    with sb4:
        st.markdown(
            f'<div style="{box_style}">'
            f'<span style="color:#6c757d; font-size:0.7rem; font-family:JetBrains Mono;">VIX Δ</span>'
            f'<span style="font-family:JetBrains Mono; font-size:1.1rem; color:{vix_color};">'
            f'{vix_change:+.1f} ({vix_change/vix_level*100:+.0f}%)</span></div>', unsafe_allow_html=True)
    
    # ── Position type and DTE filter ──
    fc_pos_col1, fc_pos_col2, fc_pos_col3 = st.columns([1, 1, 1])
    with fc_pos_col1:
        fc_option_choice = st.selectbox("Position", list(OPTION_LABELS.keys()), index=0, key="fc_pos")
        fc_opt_type, fc_opt_position = OPTION_LABELS[fc_option_choice]
    with fc_pos_col2:
        fc_min_dte = st.slider("Min DTE", 7, 180, 7, 1, key="fc_min_dte",
                                help="Mindest-Restlaufzeit — filtert zu kurze Zeiträume aus")
    with fc_pos_col3:
        fc_max_dte = st.slider("Max DTE", 30, 730, 365, 5, key="fc_max_dte",
                                help="Maximale Restlaufzeit")
    
    # ── Run quick analysis ──
    fc_df = quick_forecast_analysis(
        S, risk_free, vix_level, fc_spot_change, fc_new_vix,
        fc_opt_type, fc_opt_position, fc_min_dte, fc_max_dte
    )
    
    # Filter by DTE range and minimum premium
    fc_ranked = fc_df[(fc_df["Entry $"] >= 1.0) & 
                      (fc_df["DTE"] >= fc_min_dte) & 
                      (fc_df["DTE"] <= fc_max_dte)].copy()
    
    # ── Sweet Spot Summary ──
    st.markdown("### 🏆 Sweet Spot — Top Empfehlungen")
    
    if not fc_ranked.empty and fc_ranked["P&L %"].max() > 0:
        # --- Scoring: find practically best trades, not just max ROI% ---
        # "Best ROI" = best among options with meaningful premium (≥$5)
        # This avoids penny-options dominating the ranking
        practical = fc_ranked[fc_ranked["Entry $"] >= 5.0]
        if practical.empty:
            practical = fc_ranked  # fallback
        
        best_pct = practical.loc[practical["P&L %"].idxmax()]
        
        # Best by absolute P&L (for bigger positions)
        best_abs = fc_ranked.loc[fc_ranked["P&L $"].idxmax()]
        
        # Best risk-adjusted: P&L$ per $100 invested, with minimum $10 entry
        # This balances ROI% and absolute gains
        practical_mid = fc_ranked[fc_ranked["Entry $"] >= 10.0]
        if practical_mid.empty:
            practical_mid = fc_ranked
        best_efficiency = practical_mid.loc[practical_mid["P&L %"].idxmax()]
        
        rc1, rc2, rc3 = st.columns(3)
        
        with rc1:
            st.markdown(
                f'<div style="background:linear-gradient(145deg,#e8f5e9,#c8e6c9); border:1px solid #a5d6a7; border-radius:12px; padding:16px;">'
                f'<span style="color:#1b5e20; font-family:JetBrains Mono; font-size:0.75rem;">🥇 BESTER ROI (Entry≥$5)</span><br>'
                f'<span style="font-family:JetBrains Mono; font-size:1.6rem; color:#2e7d32; font-weight:700;">{best_pct["P&L %"]:+,.0f}%</span>'
                f'<span style="font-family:JetBrains Mono; font-size:1rem; color:#388e3c; margin-left:8px;">(${best_pct["P&L $"]:+,.0f})</span><br>'
                f'<span style="color:#333; font-family:JetBrains Mono; font-size:0.9rem;">'
                f'Strike {best_pct["Strike"]} · {best_pct["DTE"]} DTE</span><br>'
                f'<span style="color:#666; font-size:0.8rem;">Entry ${best_pct["Entry $"]:.2f} → Exit ${best_pct["Exit $"]:.2f}</span>'
                f'</div>', unsafe_allow_html=True
            )
        
        with rc2:
            st.markdown(
                f'<div style="background:linear-gradient(145deg,#e3f2fd,#bbdefb); border:1px solid #90caf9; border-radius:12px; padding:16px;">'
                f'<span style="color:#1565c0; font-family:JetBrains Mono; font-size:0.75rem;">💰 BESTER ABS. P&L</span><br>'
                f'<span style="font-family:JetBrains Mono; font-size:1.6rem; color:#1565c0; font-weight:700;">${best_abs["P&L $"]:+,.0f}</span>'
                f'<span style="font-family:JetBrains Mono; font-size:1rem; color:#1976d2; margin-left:8px;">({best_abs["P&L %"]:+,.0f}%)</span><br>'
                f'<span style="color:#333; font-family:JetBrains Mono; font-size:0.9rem;">'
                f'Strike {best_abs["Strike"]} · {best_abs["DTE"]} DTE</span><br>'
                f'<span style="color:#666; font-size:0.8rem;">Entry ${best_abs["Entry $"]:.2f} → Exit ${best_abs["Exit $"]:.2f}</span>'
                f'</div>', unsafe_allow_html=True
            )
        
        with rc3:
            st.markdown(
                f'<div style="background:linear-gradient(145deg,#f3e5f5,#e1bee7); border:1px solid #ce93d8; border-radius:12px; padding:16px;">'
                f'<span style="color:#7b1fa2; font-family:JetBrains Mono; font-size:0.75rem;">⚡ BESTE EFFIZIENZ (Entry≥$10)</span><br>'
                f'<span style="font-family:JetBrains Mono; font-size:1.6rem; color:#7b1fa2; font-weight:700;">{best_efficiency["P&L %"]:+,.0f}%</span>'
                f'<span style="font-family:JetBrains Mono; font-size:1rem; color:#9c27b0; margin-left:8px;">(${best_efficiency["P&L $"]:+,.0f})</span><br>'
                f'<span style="color:#333; font-family:JetBrains Mono; font-size:0.9rem;">'
                f'Strike {best_efficiency["Strike"]} · {best_efficiency["DTE"]} DTE</span><br>'
                f'<span style="color:#666; font-size:0.8rem;">Entry ${best_efficiency["Entry $"]:.2f} → Exit ${best_efficiency["Exit $"]:.2f}</span>'
                f'</div>', unsafe_allow_html=True
            )
        
        st.markdown("")
        
        # ── Compact Heatmap: P&L% by Strike × DTE ──
        st.markdown("#### Forecast Heatmap — P&L % (Strike × DTE)")
        
        # Use full dataset (not filtered by min premium) for complete heatmap
        fc_hm_data = fc_df[(fc_df["DTE"] >= fc_min_dte) & (fc_df["DTE"] <= fc_max_dte)].copy()
        fc_pivot = fc_hm_data.pivot_table(index="Strike", columns="DTE", values="P&L %", aggfunc="first")
        fc_pivot = fc_pivot.sort_index(ascending=False)
        
        # Format text: compact large numbers with % suffix
        def fmt_heatmap_val(v):
            if pd.isna(v):
                return ""
            if abs(v) >= 10000:
                return f"{v/1000:.0f}k%"
            elif abs(v) >= 1000:
                return f"{v/1000:.1f}k%"
            elif abs(v) < 0.5:
                return "0%"
            else:
                return f"{v:.0f}%"
        
        text_matrix = np.vectorize(fmt_heatmap_val)(fc_pivot.values)
        
        # Replace NaN with 0 for color scale
        z_values = fc_pivot.fillna(0).values
        
        # Cap color scale at 90th percentile so extreme values don't wash out the rest
        flat_vals = z_values[z_values != 0]
        if len(flat_vals) > 0:
            pos_vals = flat_vals[flat_vals > 0]
            neg_vals = flat_vals[flat_vals < 0]
            z_cap = np.percentile(pos_vals, 90) if len(pos_vals) > 0 else 100
            z_floor = np.percentile(neg_vals, 10) if len(neg_vals) > 0 else 0
        else:
            z_cap = 100
            z_floor = 0
        
        # Build custom colorscale: red below 0, white/neutral at 0, green above
        # Normalize 0-point within [z_floor, z_cap] range
        if z_cap > z_floor:
            zero_frac = abs(z_floor) / (abs(z_floor) + z_cap) if z_floor < 0 else 0.0
        else:
            zero_frac = 0.0
        
        # If all positive, use beige-to-green gradient
        if z_floor >= 0:
            custom_colorscale = [
                [0.0, "#f5f0e1"],      # warm beige (low values)
                [0.25, "#d4e6b5"],     # light sage
                [0.5, "#8bc34a"],      # medium green
                [0.75, "#4caf50"],     # strong green
                [1.0, "#1b5e20"],      # deep green
            ]
            z_floor = 0
        else:
            # Mixed: red → beige → green
            custom_colorscale = [
                [0.0, "#c62828"],
                [max(zero_frac - 0.01, 0), "#ef9a9a"],
                [zero_frac, "#f5f0e1"],
                [min(zero_frac + 0.01, 1), "#d4e6b5"],
                [min(zero_frac + (1 - zero_frac) * 0.5, 1), "#66bb6a"],
                [1.0, "#1b5e20"],
            ]
        
        fig_fc_hm = go.Figure(data=go.Heatmap(
            z=z_values,
            x=[str(c) for c in fc_pivot.columns],
            y=[str(r) for r in fc_pivot.index],
            colorscale=custom_colorscale,
            zmin=z_floor,
            zmax=z_cap,
            text=text_matrix,
            texttemplate="%{text}",
            textfont={"size": 9, "family": "JetBrains Mono"},
            hovertemplate="Strike: %{y}<br>DTE: %{x}<br>P&L: %{z:,.1f}%<extra></extra>",
            colorbar=dict(title=dict(text="P&L %", font=dict(family="JetBrains Mono"))),
        ))
        
        fig_fc_hm.update_layout(
            xaxis_title="DTE",
            yaxis_title="Strike",
            font=dict(family="JetBrains Mono, monospace"),
            height=max(350, len(fc_pivot.index) * 20),
            template="plotly_white",
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(248,249,250,1)",
            margin=dict(l=80, r=40, t=20, b=60),
        )
        
        st.plotly_chart(fig_fc_hm, use_container_width=True)
        
        # ── P&L curve by Spot price for top strikes ──
        st.markdown("#### P&L % nach Kurs — Top Strikes")
        
        # Pick top strikes from the practical set (Entry >= $5)
        practical_for_chart = fc_ranked[fc_ranked["Entry $"] >= 5.0]
        if practical_for_chart.empty:
            practical_for_chart = fc_ranked
        
        # Get best strike per DTE bucket, then pick top 5 unique strikes
        top_strikes_chart = (practical_for_chart.groupby("Strike")["P&L %"].max()
                             .nlargest(5).index.tolist())
        
        # Use a mid-range DTE for the spot scan (pick most common DTE in top results)
        best_dte_for_chart = int(practical_for_chart.loc[
            practical_for_chart["Strike"].isin(top_strikes_chart), "DTE"
        ].mode().iloc[0]) if not practical_for_chart.empty else 60
        
        # Simulate P&L across spot prices for each top strike
        spot_range_pct = np.linspace(-25, 10, 71)  # -25% to +10%
        
        fig_fc_spot = go.Figure()
        colors_fc = ["#00e676", "#42a5f5", "#ffa726", "#ab47bc", "#ef5350"]
        
        current_iv_fc = vix_level / 100
        new_iv_fc = fc_new_vix / 100
        
        for i, K in enumerate(sorted(top_strikes_chart)):
            T_entry = best_dte_for_chart / 365
            T_exit = max((best_dte_for_chart - 1) / 365, 1/365)
            entry_p = bs_price(S, K, T_entry, risk_free, current_iv_fc, fc_opt_type)
            
            if entry_p < 0.01:
                continue
            
            pnl_curve = []
            for sp_pct in spot_range_pct:
                new_spot_sim = S * (1 + sp_pct / 100)
                # Scale IV with spot move (auto-estimate)
                sim_vix = estimate_vix_from_spot_change(sp_pct, vix_level)
                sim_iv = sim_vix / 100
                exit_p = bs_price(new_spot_sim, K, T_exit, risk_free, sim_iv, fc_opt_type)
                if fc_opt_position == "long":
                    pnl = (exit_p - entry_p) / entry_p * 100
                else:
                    pnl = (entry_p - exit_p) / entry_p * 100
                pnl_curve.append(pnl)
            
            moneyness = (K / S - 1) * 100
            fig_fc_spot.add_trace(go.Scatter(
                x=spot_range_pct, y=pnl_curve,
                mode="lines",
                name=f"{K} ({moneyness:+.1f}%) {best_dte_for_chart}D",
                line=dict(width=2.5, color=colors_fc[i % len(colors_fc)]),
            ))
        
        # Mark current forecast
        fig_fc_spot.add_vline(x=fc_spot_change, line_dash="dash", line_color="#e65100",
                              opacity=0.6, annotation_text=f"Prognose {fc_spot_change:+.1f}%",
                              annotation_font_color="#e65100", annotation_font_size=10)
        fig_fc_spot.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.3)
        
        fig_fc_spot.update_layout(
            xaxis_title="Spot-Veränderung (%)",
            yaxis_title="P&L %",
            font=dict(family="JetBrains Mono, monospace"),
            height=380,
            template="plotly_white",
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(248,249,250,1)",
            legend=dict(font=dict(size=11)),
            margin=dict(l=60, r=40, t=20, b=60),
        )
        st.plotly_chart(fig_fc_spot, use_container_width=True)
        
        # ── Top 15 table (same $5 filter as recommendations) ──
        st.markdown("#### Top 15 Trades (Entry ≥ $5)")
        top15_practical = fc_ranked[fc_ranked["Entry $"] >= 5.0].nlargest(15, "P&L %")
        if top15_practical.empty:
            top15_practical = fc_ranked.nlargest(15, "P&L %")
        st.dataframe(top15_practical.reset_index(drop=True), use_container_width=True, hide_index=True)
        
    else:
        st.warning("Kein profitabler Trade bei dieser Prognose gefunden. Passe Spot-/VIX-Prognose an.")

# ─── TAB 1: HEATMAP ──────────────────────────────────
with tab_heat:
    st.markdown("### P&L Heatmap: Strike × DTE")
    
    heatmap_metric = st.radio("Metric", ["PnL", "PnL_%", "Vega", "Delta"], horizontal=True, key="hm_metric")
    
    # Pivot for heatmap
    pivot = df.pivot_table(index="Strike", columns="DTE", values=heatmap_metric, aggfunc="first")
    pivot = pivot.sort_index(ascending=False)
    
    # Color scale
    if heatmap_metric in ["PnL", "PnL_%"]:
        if opt_position == "long":
            colorscale = "RdYlGn"
        else:
            colorscale = "RdYlGn"
        zmid = 0
    elif heatmap_metric == "Vega":
        colorscale = "Viridis"
        zmid = None
    else:
        colorscale = "RdBu"
        zmid = 0
    
    fig_hm = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=[str(r) for r in pivot.index],
        colorscale=colorscale,
        zmid=zmid,
        text=np.round(pivot.values, 1),
        texttemplate="%{text}",
        textfont={"size": 9, "family": "JetBrains Mono"},
        hovertemplate="Strike: %{y}<br>DTE: %{x}<br>Value: %{z:.2f}<extra></extra>",
        colorbar=dict(title=dict(text=heatmap_metric, font=dict(family="JetBrains Mono"))),
    ))
    
    fig_hm.update_layout(
        xaxis_title="DTE",
        yaxis_title="Strike",
        font=dict(family="JetBrains Mono, monospace"),
        height=max(400, len(K_range) * 22),
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(248,249,250,1)",
        margin=dict(l=80, r=40, t=40, b=60),
    )
    
    # Mark ATM strike
    atm_strike = min(K_range, key=lambda k: abs(k - S))
    y_labels = [str(r) for r in pivot.index]
    if str(int(atm_strike)) in y_labels:
        atm_idx = y_labels.index(str(int(atm_strike)))
        fig_hm.add_shape(
            type="line", x0=-0.5, x1=len(pivot.columns)-0.5,
            y0=atm_idx, y1=atm_idx,
            line=dict(color="white", width=1.5, dash="dot"), opacity=0.6,
        )
        fig_hm.add_annotation(
            x=-0.5, y=atm_idx, text="ATM", showarrow=False,
            font=dict(color="#333", size=10, family="JetBrains Mono"),
            xanchor="right", xshift=-5,
        )
    
    st.plotly_chart(fig_hm, use_container_width=True)
    
    # Best combos
    st.markdown("#### 🏆 Top 10 Best Combinations")
    if opt_position == "long":
        best = df.nlargest(10, "PnL_%")
    else:
        best = df.nlargest(10, "PnL_%")
    
    st.dataframe(
        best[["Strike", "DTE", "Moneyness_%", "Entry_Price", "Exit_Price", 
              "PnL", "PnL_%", "Delta", "Vega", "Theta"]].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

# ─── TAB 2: RESULTS TABLE ──────────────────────────────────
with tab_table:
    st.markdown("### Full Results Grid")
    
    # Filters
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filter_dte = st.multiselect("Filter DTE", sorted(df["DTE"].unique()), 
                                     default=sorted(df["DTE"].unique())[:8])
    with col_f2:
        sort_by = st.selectbox("Sort by", ["PnL_%", "PnL", "Vega", "Entry_Price", "Strike", "DTE"])
    
    filtered = df[df["DTE"].isin(filter_dte)].sort_values(sort_by, ascending=False)
    
    # Color the PnL columns
    def highlight_pnl(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return "color: #00e676"
            elif val < 0:
                return "color: #ff5252"
        return ""
    
    styled = filtered.style.applymap(highlight_pnl, subset=["PnL", "PnL_%"])
    st.dataframe(filtered, use_container_width=True, hide_index=True, height=500)

# ─── TAB 3: DTE CURVES ──────────────────────────────────
with tab_curves:
    st.markdown("### P&L by DTE for Selected Strikes")
    
    # Typical DTE buckets
    typical_dtes = [7, 14, 30, 60, 90, 120, 180]
    available_dtes = sorted(df["DTE"].unique())
    
    selected_strikes = st.multiselect(
        "Select Strikes to compare",
        sorted(df["Strike"].unique()),
        default=sorted(df["Strike"].unique(), key=lambda k: abs(k - S * 0.95))[:5],
        key="curve_strikes"
    )
    
    if selected_strikes:
        # P&L% vs DTE
        fig_curves = go.Figure()
        colors = px.colors.qualitative.Set2
        
        for i, K in enumerate(sorted(selected_strikes)):
            subset = df[df["Strike"] == K].sort_values("DTE")
            moneyness = (K / S - 1) * 100
            color = colors[i % len(colors)]
            
            fig_curves.add_trace(go.Scatter(
                x=subset["DTE"], y=subset["PnL_%"],
                mode="lines+markers",
                name=f"{int(K)} ({moneyness:+.1f}%)",
                line=dict(width=2.5, color=color),
                marker=dict(size=5),
                hovertemplate=f"Strike {int(K)}<br>DTE: %{{x}}<br>P&L: %{{y:.1f}}%<br>Entry: $%{{customdata[0]:.2f}}<extra></extra>",
                customdata=subset[["Entry_Price"]].values,
            ))
        
        fig_curves.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_curves.update_layout(
            xaxis_title="DTE",
            yaxis_title="P&L %",
            title="Scenario P&L% by DTE",
            font=dict(family="JetBrains Mono, monospace"),
            height=450,
            template="plotly_white",
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(248,249,250,1)",
            legend=dict(font=dict(size=11)),
            margin=dict(l=60, r=40, t=60, b=60),
        )
        st.plotly_chart(fig_curves, use_container_width=True)
        
        # Entry price vs DTE
        fig_entry = go.Figure()
        for i, K in enumerate(sorted(selected_strikes)):
            subset = df[df["Strike"] == K].sort_values("DTE")
            color = colors[i % len(colors)]
            fig_entry.add_trace(go.Scatter(
                x=subset["DTE"], y=subset["Entry_Price"],
                mode="lines+markers",
                name=f"{int(K)}",
                line=dict(width=2, color=color),
                marker=dict(size=4),
            ))
        
        fig_entry.update_layout(
            xaxis_title="DTE",
            yaxis_title="Entry Price ($)",
            title="Entry Cost by DTE",
            font=dict(family="JetBrains Mono, monospace"),
            height=350,
            template="plotly_white",
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(248,249,250,1)",
            margin=dict(l=60, r=40, t=60, b=60),
        )
        st.plotly_chart(fig_entry, use_container_width=True)
        
        # Vega exposure by DTE
        fig_vega = go.Figure()
        for i, K in enumerate(sorted(selected_strikes)):
            subset = df[df["Strike"] == K].sort_values("DTE")
            color = colors[i % len(colors)]
            fig_vega.add_trace(go.Bar(
                x=subset["DTE"], y=subset["Vega"],
                name=f"{int(K)}",
                marker_color=color,
                opacity=0.8,
            ))
        
        fig_vega.update_layout(
            xaxis_title="DTE",
            yaxis_title="Vega (per 1% IV)",
            title="Vega Exposure by DTE",
            font=dict(family="JetBrains Mono, monospace"),
            height=350,
            template="plotly_white",
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(248,249,250,1)",
            barmode="group",
            margin=dict(l=60, r=40, t=60, b=60),
        )
        st.plotly_chart(fig_vega, use_container_width=True)
    else:
        st.info("Select at least one strike to see curves.")

# ─── TAB 4: MULTI-DAY SIMULATION ──────────────────────────────────
with tab_sim:
    st.markdown("### Multi-Day Scenario Simulation")
    st.markdown("Simulate how a position evolves over several days with gradual spot & vol changes.")
    
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        sim_strike = st.number_input("Strike", value=int(round(S * 0.95 / strike_step) * strike_step),
                                      step=strike_step, key="sim_strike")
    with col_s2:
        sim_dte = st.number_input("DTE at Entry", value=30, min_value=5, max_value=365, step=1, key="sim_dte")
    with col_s3:
        sim_days = st.number_input("Simulation Days", value=10, min_value=2, max_value=60, step=1, key="sim_days")
    
    # Spot path presets
    st.markdown("**Spot Path**")
    path_type = st.selectbox("Path Type", [
        "Linear decline", "Sudden crash (day 3)", "V-recovery", "Slow grind down", 
        "Flat", "Linear rally", "Custom"
    ])
    
    total_move = spot_change  # use sidebar value
    days = sim_days
    
    if path_type == "Linear decline":
        spot_path = np.linspace(0, total_move, days + 1)
    elif path_type == "Sudden crash (day 3)":
        spot_path = np.zeros(days + 1)
        spot_path[3:] = total_move
    elif path_type == "V-recovery":
        mid = days // 2
        spot_path = np.concatenate([
            np.linspace(0, total_move, mid + 1),
            np.linspace(total_move, total_move * 0.3, days - mid)
        ])
    elif path_type == "Slow grind down":
        spot_path = total_move * (1 - np.exp(-np.linspace(0, 3, days + 1)))
        spot_path = spot_path / spot_path[-1] * total_move if spot_path[-1] != 0 else np.zeros(days+1)
    elif path_type == "Flat":
        spot_path = np.zeros(days + 1)
    elif path_type == "Linear rally":
        spot_path = np.linspace(0, abs(total_move), days + 1)
    else:  # Custom
        custom_str = st.text_input("Custom path (comma-separated % per day)", 
                                    value=",".join([str(round(x, 1)) for x in np.linspace(0, total_move, days + 1)]))
        try:
            spot_path = np.array([float(x.strip()) for x in custom_str.split(",")])
        except:
            spot_path = np.linspace(0, total_move, days + 1)
    
    # Vol change
    vol_change_total = (new_iv - current_iv)
    
    # Run simulation
    sim_df = multi_day_scenario(
        S, sim_strike, sim_dte, risk_free, current_iv,
        vol_change_total, spot_path, opt_type, opt_position, days
    )
    
    # Plot: Combined chart
    fig_sim = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("P&L ($) and P&L (%)", "Spot Price & IV", "Greeks Evolution"),
        row_heights=[0.4, 0.3, 0.3],
    )
    
    # P&L
    colors_pnl = ["#00e676" if v >= 0 else "#ff5252" for v in sim_df["PnL"]]
    fig_sim.add_trace(go.Bar(
        x=sim_df["Day"], y=sim_df["PnL"],
        name="P&L $", marker_color=colors_pnl, opacity=0.7,
    ), row=1, col=1)
    
    fig_sim.add_trace(go.Scatter(
        x=sim_df["Day"], y=sim_df["PnL_%"],
        name="P&L %", mode="lines+markers",
        line=dict(color="#ffa726", width=2.5),
        yaxis="y2",
    ), row=1, col=1)
    
    # Spot & IV
    fig_sim.add_trace(go.Scatter(
        x=sim_df["Day"], y=sim_df["Spot"],
        name="Spot", mode="lines+markers",
        line=dict(color="#42a5f5", width=2),
    ), row=2, col=1)
    
    fig_sim.add_trace(go.Scatter(
        x=sim_df["Day"], y=sim_df["IV"],
        name="IV %", mode="lines+markers",
        line=dict(color="#ef5350", width=2, dash="dot"),
    ), row=2, col=1)
    
    # Greeks
    fig_sim.add_trace(go.Scatter(
        x=sim_df["Day"], y=sim_df["Delta"],
        name="Delta", mode="lines",
        line=dict(color="#66bb6a", width=2),
    ), row=3, col=1)
    
    fig_sim.add_trace(go.Scatter(
        x=sim_df["Day"], y=sim_df["Vega"],
        name="Vega", mode="lines",
        line=dict(color="#ab47bc", width=2),
    ), row=3, col=1)
    
    fig_sim.add_trace(go.Scatter(
        x=sim_df["Day"], y=sim_df["Theta"],
        name="Theta", mode="lines",
        line=dict(color="#ff7043", width=2),
    ), row=3, col=1)
    
    fig_sim.update_layout(
        height=750,
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(248,249,250,1)",
        font=dict(family="JetBrains Mono, monospace", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=80, b=40),
        showlegend=True,
    )
    
    fig_sim.update_xaxes(title_text="Day", row=3, col=1)
    
    st.plotly_chart(fig_sim, use_container_width=True)
    
    # Summary table
    st.markdown("#### Simulation Data")
    st.dataframe(sim_df, use_container_width=True, hide_index=True)

# ─── Footer ──────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#555; font-size:0.8rem; font-family:'JetBrains Mono',monospace;">
    Vol Scenario Analyzer · Black-Scholes · Market data via Yahoo Finance<br>
    ⚠️ For analysis purposes only — not financial advice
</div>
""", unsafe_allow_html=True)
