import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dictionary mapping Nifty 50 stocks to their sectors
nifty_50_stocks = {
    'RELIANCE.NS': 'Energy', 'TCS.NS': 'Technology', 'INFY.NS': 'Technology',
    'HDFCBANK.NS': 'Financials', 'ICICIBANK.NS': 'Financials', 'KOTAKBANK.NS': 'Financials',
    'HINDUNILVR.NS': 'Consumer Goods', 'ITC.NS': 'Consumer Goods', 'SBIN.NS': 'Financials',
    'BHARTIARTL.NS': 'Telecom', 'ASIANPAINT.NS': 'Consumer Goods', 'BAJFINANCE.NS': 'Financials',
    'AXISBANK.NS': 'Financials', 'HCLTECH.NS': 'Technology', 'MARUTI.NS': 'Automobile',
    'ULTRACEMCO.NS': 'Cement', 'LT.NS': 'Infrastructure', 'NESTLEIND.NS': 'Consumer Goods',
    'SUNPHARMA.NS': 'Pharmaceuticals', 'DRREDDY.NS': 'Pharmaceuticals'
}

# Streamlit UI
st.title("📊 Nifty 50 Sector-Wise Trend Analyzer")
selected_sector = st.sidebar.selectbox("Select Sector", sorted(set(nifty_50_stocks.values())))
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"])
rsi_period = st.sidebar.slider("RSI Period", min_value=5, max_value=30, value=14)

# Filter stocks by selected sector
filtered_stocks = [ticker for ticker, sector in nifty_50_stocks.items() if sector == selected_sector]

# Fetch data
@st.cache_data
def fetch_data(tickers, interval):
    return yf.download(tickers, period="6mo", interval=interval, group_by="ticker", auto_adjust=True)

data = fetch_data(filtered_stocks, interval)

# RSI calculation function
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Plot RSI for each stock
st.subheader(f"📈 RSI for {selected_sector} Sector Stocks")
for ticker in filtered_stocks:
    try:
        price_series = data[ticker]['Close']
        rsi = calculate_rsi(price_series, period=rsi_period)

        fig, ax = plt.subplots(figsize=(7, 2.5))
        ax.plot(rsi, label=f"{ticker} RSI", color='blue')
        ax.axhline(70, color='red', linestyle='--', linewidth=1)
        ax.axhline(30, color='green', linestyle='--', linewidth=1)
        ax.set_title(f"{ticker} - RSI")
        ax.set_ylabel("RSI")
        ax.set_xlabel("Date")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"Could not calculate RSI for {ticker}: {e}")
