import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# -----------------------------
# Market Data
# -----------------------------
def get_stock_data(ticker):
    data = yf.download(ticker, period="1y", interval="1d", progress=False)
    if data.empty:
        raise ValueError("No data found.")

    prices = data["Close"]
    S0 = float(prices.iloc[-1])
    log_returns = np.log(prices / prices.shift(1)).dropna()
    sigma = float(log_returns.std() * np.sqrt(252))

    return S0, sigma


# -----------------------------
# European Option (India)
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
# American Option (US - LSM)
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
# MAIN PROGRAM
# -----------------------------
exchange = input("Choose exchange (US / NSE / BSE): ").upper()
ticker_input = input("Enter stock ticker: ").upper()
option_type = input("Option type (call / put): ").lower()

if exchange == "US":
    ticker = ticker_input
    r = 0.05
    model = "American"

elif exchange == "NSE":
    ticker = ticker_input + ".NS"
    r = 0.06
    model = "European"

elif exchange == "BSE":
    ticker = ticker_input + ".BO"
    r = 0.06
    model = "European"

else:
    raise ValueError("Invalid exchange")

print(f"\nFetching data for {ticker}...")
S0, sigma = get_stock_data(ticker)

print(f"Spot Price: {S0:.2f}")
print(f"Volatility: {sigma:.4f}")
print(f"Model Used: {model} {option_type.upper()}")

K = float(input("Enter strike price: "))
expiry_str = input("Enter expiry date (DD-MM-YYYY): ")

expiry_date = datetime.strptime(expiry_str, "%d-%m-%Y")
today = datetime.now()
T = (expiry_date - today).days / 365

if T <= 0:
    raise ValueError("Expiry date must be in the future.")

print(f"Time to Expiry: {T:.4f} years ({(expiry_date - today).days} days)")

if model == "American":
    price, ST, payoff = american_option_lsm(S0, K, r, sigma, T=T, option_type=option_type)
else:
    price, ST, payoff = european_option_mc(S0, K, r, sigma, T=T, option_type=option_type)

print(f"\nOption Price for {ticker}: {price:.4f}")

# -----------------------------
# Plot Results
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Simulated Stock Prices at Expiry
axes[0].hist(ST, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(S0, color='green', linestyle='--', linewidth=2, label=f'Spot Price: {S0:.2f}')
axes[0].axvline(K, color='red', linestyle='--', linewidth=2, label=f'Strike Price: {K:.2f}')
axes[0].set_xlabel('Stock Price at Expiry', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title(f'{ticker} - Simulated Stock Prices at Expiry', fontsize=14)
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

plt.suptitle(f'{model} {option_type.upper()} Option - Strike: {K}, T: {T:.4f} years', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
