# 📈 Monte Carlo Options Pricing

A sophisticated options pricing tool using Monte Carlo simulation methods. Supports both **American** (Longstaff-Schwartz Method) and **European** options pricing with real-time market data.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.52-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

- **Real-time Market Data**: Fetches live stock prices and calculates historical volatility using Yahoo Finance
- **Dual Pricing Models**:
  - **American Options (LSM)**: Longstaff-Schwartz Least Squares Monte Carlo for US stocks
  - **European Options (MC)**: Standard Monte Carlo simulation for Indian stocks (NSE/BSE)
- **Interactive Web App**: Beautiful Streamlit interface with real-time calculations
- **Visualizations**: Distribution charts for simulated prices and option payoffs
- **Multi-Exchange Support**: US, NSE (India), and BSE (India)

## 🖥️ Screenshots

The app features a premium gradient UI with:
- Interactive sidebar for configuration
- Real-time stock data display
- Monte Carlo simulation visualization
- Key statistics (ITM%, Mean Payoff, Max Payoff)

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy yfinance matplotlib streamlit
```

### Run the Streamlit App

```bash
streamlit run app.py
```

### Run the CLI Version

```bash
python "Options Simulation.py"
```

## 📊 How It Works

### Monte Carlo Simulation

The tool simulates thousands of possible stock price paths using Geometric Brownian Motion:

```
S(T) = S(0) × exp[(r - σ²/2)T + σ√T × Z]
```

Where:
- `S(0)` = Current stock price
- `r` = Risk-free rate
- `σ` = Volatility (annualized)
- `T` = Time to expiry
- `Z` = Standard normal random variable

### Longstaff-Schwartz Method (American Options)

For American options, the LSM algorithm determines optimal early exercise by:
1. Simulating price paths
2. Working backwards from expiry
3. Using regression to estimate continuation value
4. Comparing exercise value vs. continuation value

## 🔧 Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| Exchange | US / NSE / BSE | - |
| Ticker | Stock symbol | - |
| Option Type | Call / Put | Put |
| Strike Price | Exercise price | - |
| Expiry Date | Option expiration | - |
| Simulations | Number of Monte Carlo paths | 50,000 |

### Risk-Free Rates
- **US**: 5% (Fed rate)
- **India (NSE/BSE)**: 6% (RBI rate)

## 📁 Project Structure

```
Monte Carlo Simulation/
├── app.py                  # Streamlit web application
├── Options Simulation.py   # Command-line interface
├── README.md              # This file
└── requirements.txt       # Dependencies
```

## 🛠️ Technologies Used

- **Python 3.10+**
- **NumPy** - Numerical computations
- **yfinance** - Market data API
- **Matplotlib** - Visualizations
- **Streamlit** - Web application framework

## 📈 Example Output

```
Fetching data for AAPL...
Spot Price: 250.42
Volatility: 0.2534
Model Used: American CALL
Time to Expiry: 0.0164 years (6 days)

Option Price for AAPL: 23.8093
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**SMKehTDOG2335**

- GitHub: [@SMKehTDOG2335](https://github.com/SMKehTDOG2335)

---

⭐ Star this repo if you found it helpful!
