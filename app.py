"""
Monte Carlo Options Pricing - Streamlit Web Application
=========================================================
Author: Steve Mathews Korah
GitHub: https://github.com/SMKehTDOG2335
Copyright (c) 2026 Steve Mathews Korah. All rights reserved.

A sophisticated options pricing tool using Monte Carlo simulation methods.
Supports both American (Longstaff-Schwartz) and European options pricing.
"""

import streamlit as st
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Monte Carlo Options Pricing",
    page_icon="📈",
    layout="wide"
)

# -----------------------------
# Custom CSS for Premium Look
# -----------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 0.5rem;
        width: 100%;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Market Data Function
# -----------------------------
@st.cache_data(ttl=300)
def get_stock_data(ticker):
    data = yf.download(ticker, period="1y", interval="1d", progress=False)
    if data.empty:
        return None, None
    
    prices = data["Close"]
    S0 = float(prices.iloc[-1])
    log_returns = np.log(prices / prices.shift(1)).dropna()
    sigma = float(log_returns.std() * np.sqrt(252))
    
    return S0, sigma

# -----------------------------
# European Option (MC)
# -----------------------------
def european_option_mc(S0, K, r, sigma, T=1.0, N=50000, option_type="put"):
    Z = np.random.normal(size=N)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    if option_type == "call":
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)
    
    price = np.exp(-r * T) * np.mean(payoff)
    return price, ST, payoff

# -----------------------------
# American Option (LSM)
# -----------------------------
def american_option_lsm(S0, K, r, sigma, T=1.0, M=50, N=20000, option_type="put"):
    dt = T / M
    Z = np.random.normal(size=(N, M))
    S = np.zeros((N, M + 1))
    S[:, 0] = S0
    
    for t in range(1, M + 1):
        S[:, t] = S[:, t - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt +
            sigma * np.sqrt(dt) * Z[:, t - 1]
        )
    
    if option_type == "call":
        payoff = np.maximum(S - K, 0)
    else:
        payoff = np.maximum(K - S, 0)
    
    V = payoff[:, -1]
    discount = np.exp(-r * dt)
    
    for t in range(M - 1, 0, -1):
        itm = payoff[:, t] > 0
        if not np.any(itm):
            V *= discount
            continue
        
        S_itm = S[itm, t]
        X = np.column_stack([np.ones(len(S_itm)), S_itm, S_itm**2])
        Y = V[itm] * discount
        
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        continuation = X @ beta
        exercise = payoff[itm, t]
        
        V[itm] = np.where(exercise > continuation, exercise, Y)
        V[~itm] *= discount
    
    price = np.mean(V) * np.exp(-r * dt)
    ST = S[:, -1]
    final_payoff = payoff[:, -1]
    return price, ST, final_payoff

# -----------------------------
# Main App
# -----------------------------
st.markdown('<h1 class="main-header">📈 Monte Carlo Options Pricing</h1>', unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.header("⚙️ Configuration")
    
    exchange = st.selectbox(
        "Exchange",
        ["US", "NSE", "BSE"],
        help="US: American Options, NSE/BSE: European Options"
    )
    
    ticker_input = st.text_input(
        "Stock Ticker",
        value="AAPL" if exchange == "US" else "RELIANCE",
        help="Enter the stock symbol"
    ).upper()
    
    option_type = st.radio(
        "Option Type",
        ["Call", "Put"],
        horizontal=True
    ).lower()
    
    st.divider()
    
    strike_price = st.number_input(
        "Strike Price",
        min_value=0.01,
        value=100.0,
        step=1.0,
        help="The strike price of the option"
    )
    
    expiry_date = st.date_input(
        "Expiry Date",
        value=datetime.now() + timedelta(days=30),
        min_value=datetime.now().date() + timedelta(days=1),
        help="Option expiration date"
    )
    
    st.divider()
    
    num_simulations = st.slider(
        "Number of Simulations",
        min_value=10000,
        max_value=100000,
        value=50000,
        step=10000,
        help="More simulations = more accuracy but slower"
    )
    
    calculate_btn = st.button("🚀 Calculate Option Price", type="primary")

# Determine ticker and parameters
if exchange == "US":
    ticker = ticker_input
    r = 0.05
    model = "American"
elif exchange == "NSE":
    ticker = ticker_input + ".NS"
    r = 0.06
    model = "European"
else:
    ticker = ticker_input + ".BO"
    r = 0.06
    model = "European"

# Calculate time to expiry
T = (expiry_date - datetime.now().date()).days / 365

# Main content area
col1, col2, col3 = st.columns(3)

if calculate_btn and T > 0:
    with st.spinner("Fetching market data..."):
        S0, sigma = get_stock_data(ticker)
    
    if S0 is None:
        st.error(f"❌ Could not fetch data for {ticker}. Please check the ticker symbol.")
    else:
        # Display market data
        with col1:
            st.metric("Spot Price", f"${S0:.2f}" if exchange == "US" else f"₹{S0:.2f}")
        with col2:
            st.metric("Volatility (Annual)", f"{sigma*100:.2f}%")
        with col3:
            st.metric("Time to Expiry", f"{int(T*365)} days")
        
        # Calculate option price
        with st.spinner(f"Running {num_simulations:,} Monte Carlo simulations..."):
            if model == "American":
                price, ST, payoff = american_option_lsm(
                    S0, strike_price, r, sigma, T=T, N=num_simulations, option_type=option_type
                )
            else:
                price, ST, payoff = european_option_mc(
                    S0, strike_price, r, sigma, T=T, N=num_simulations, option_type=option_type
                )
        
        # Display results
        st.divider()
        
        result_col1, result_col2, result_col3 = st.columns(3)
        with result_col1:
            currency = "$" if exchange == "US" else "₹"
            st.metric("Option Price", f"{currency}{price:.4f}")
        with result_col2:
            st.metric("Model Used", f"{model} {option_type.upper()}")
        with result_col3:
            st.metric("Risk-Free Rate", f"{r*100:.1f}%")
        
        st.divider()
        
        # Plots
        st.subheader("📊 Simulation Results")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Simulated Stock Prices at Expiry
        axes[0].hist(ST, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(S0, color='green', linestyle='--', linewidth=2, label=f'Spot: {S0:.2f}')
        axes[0].axvline(strike_price, color='red', linestyle='--', linewidth=2, label=f'Strike: {strike_price:.2f}')
        axes[0].set_xlabel('Stock Price at Expiry', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title(f'{ticker} - Simulated Prices at Expiry', fontsize=14)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Plot 2: Option Payoff Distribution
        axes[1].hist(payoff, bins=100, color='coral', edgecolor='black', alpha=0.7)
        axes[1].axvline(price, color='darkred', linestyle='-', linewidth=2, label=f'Option Price: {price:.4f}')
        axes[1].set_xlabel('Payoff', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title(f'{option_type.upper()} Option Payoff Distribution', fontsize=14)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.suptitle(f'{model} {option_type.upper()} Option - Strike: {strike_price}, T: {T:.4f} years', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Statistics
        st.subheader("📈 Simulation Statistics")
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            itm_pct = np.mean(payoff > 0) * 100
            st.metric("In-The-Money %", f"{itm_pct:.1f}%")
        with stat_col2:
            st.metric("Mean Payoff", f"{currency}{np.mean(payoff):.4f}")
        with stat_col3:
            st.metric("Max Payoff", f"{currency}{np.max(payoff):.2f}")
        with stat_col4:
            st.metric("Std Dev (Prices)", f"{currency}{np.std(ST):.2f}")

elif T <= 0 and calculate_btn:
    st.error("❌ Expiry date must be in the future!")
else:
    # Show placeholder
    st.info("👈 Configure your options parameters in the sidebar and click **Calculate Option Price** to begin.")
    
    # Show example
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        1. **Select Exchange**: Choose US (American options) or NSE/BSE (European options)
        2. **Enter Ticker**: Stock symbol (e.g., AAPL, MSFT for US; RELIANCE, TCS for NSE)
        3. **Choose Option Type**: Call (right to buy) or Put (right to sell)
        4. **Set Strike Price**: The price at which you can exercise the option
        5. **Select Expiry Date**: When the option expires
        6. **Click Calculate**: Run Monte Carlo simulation to price the option
        
        **Models Used:**
        - **US Stocks**: Longstaff-Schwartz Method (LSM) for American options
        - **Indian Stocks**: Standard Monte Carlo for European options
        """)
