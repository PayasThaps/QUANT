import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import time

# Retry function for data download
def download_data_with_retry(ticker, start_date, end_date, interval, max_retries=3):
    for _ in range(max_retries):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
            if not data.empty:
                return data
        except Exception as e:
            time.sleep(1)
    return pd.DataFrame()

# Nifty 50 stocks with sectors
nifty50_stocks = {
    'RELIANCE.NS': 'Energy', 'TCS.NS': 'Technology', 'INFY.NS': 'Technology',
    'HDFCBANK.NS': 'Financials', 'ICICIBANK.NS': 'Financials', 'HINDUNILVR.NS': 'Consumer',
    'ITC.NS': 'Consumer', 'KOTAKBANK.NS': 'Financials', 'LT.NS': 'Industrials',
    'SBIN.NS': 'Financials', 'BHARTIARTL.NS': 'Telecom', 'ASIANPAINT.NS': 'Consumer',
    'AXISBANK.NS': 'Financials', 'HCLTECH.NS': 'Technology', 'MARUTI.NS': 'Automobile'
}

# App title and sidebar
st.title("Nifty 50 Sector-Wise Trend Analyzer")
selected_sector = st.sidebar.selectbox("Select Sector", sorted(set(nifty50_stocks.values())))
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"])
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))

# Sector-wise Stock Data
sector_stocks = [s for s, sector in nifty50_stocks.items() if sector == selected_sector]
stock_data_dict = {}
for stock in sector_stocks:
    data = download_data_with_retry(stock, start_date, end_date, interval)
    if not data.empty and 'Adj Close' in data.columns:
        stock_data_dict[stock] = data
    else:
        st.warning(f"Skipping {stock}: No 'Adj Close' data available.")

if not stock_data_dict:
    st.error("No valid stock data found for selected sector and time range.")
    st.stop()

# Market Benchmark (e.g., Nifty 50 index)
market_ticker = "^NSEI"
market_data = download_data_with_retry(market_ticker, start_date, end_date, interval)
if market_data.empty or 'Adj Close' not in market_data.columns:
    st.warning("Market data is not available or missing 'Adj Close'.")
    st.stop()

# Align market and stock data
market_returns = market_data['Adj Close'].pct_change().dropna()
beta_results = {}

for stock, data in stock_data_dict.items():
    stock_returns = data['Adj Close'].pct_change().dropna()
    combined = pd.concat([stock_returns, market_returns], axis=1).dropna()
    combined.columns = ['Stock', 'Market']
    
    if len(combined) > 1:
        model = LinearRegression()
        model.fit(combined[['Market']], combined['Stock'])
        beta = model.coef_[0]
        r_squared = r2_score(combined['Stock'], model.predict(combined[['Market']]))
        beta_results[stock] = {'Beta': beta, 'R-squared': r_squared}

# Display results
if beta_results:
    st.subheader(f"Beta Analysis for {selected_sector} Sector")
    beta_df = pd.DataFrame(beta_results).T
    st.dataframe(beta_df.style.format({"Beta": "{:.2f}", "R-squared": "{:.2f}"}))
else:
    st.warning("Unable to calculate beta values due to insufficient data.")
