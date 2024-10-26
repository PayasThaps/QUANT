import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go
from scipy.optimize import minimize
import time

# Function to download stock data with retries and caching using Streamlit's cache
@st.cache_data
def download_data_with_retry(ticker, start_date, end_date, interval='1d', retries=5, delay=5):
    for i in range(retries):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
            if not data.empty:
                return data
        except Exception as e:
            time.sleep(delay)
    return pd.DataFrame()  # Return empty DataFrame if all retries fail

# Function to calculate Awesome Oscillator (AO)
def awesome_oscillator(stock_data):
    median_price = (stock_data['High'] + stock_data['Low']) / 2
    ao = median_price.rolling(window=5).mean() - median_price.rolling(window=34).mean()
    return ao

# Function to calculate RSI
def calculate_rsi(stock_data, periods=14):
    delta = stock_data['Adj Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to analyze RSI
def analyze_rsi(rsi_value):
    if rsi_value > 70:
        return 'Overbought (Sell signal)', 'Bearish'
    elif rsi_value < 30:
        return 'Oversold (Buy signal)', 'Bullish'
    else:
        return 'Neutral', 'Neutral'

# Function to analyze Awesome Oscillator (AO)
def analyze_ao(ao_value):
    if np.isnan(ao_value):
        return 'Neutral', 'Hold'
    elif ao_value > 0:
        return 'Bullish', 'Buy signal'
    elif ao_value < 0:
        return 'Bearish', 'Sell signal'
    else:
        return 'Neutral', 'Hold'

# Function to calculate the Beta of a stock
def calculate_beta(stock_returns, market_returns):
    covariance_matrix = np.cov(stock_returns, market_returns)
    beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
    return beta

# List of NIFTY 50 companies and their sectors
nifty50_stocks = {
    'RELIANCE.NS': 'Energy', 'TCS.NS': 'IT', 'HDFCBANK.NS': 'Financials', 'INFY.NS': 'IT',
    # Add remaining stocks and sectors as per the provided list
}

# Parameters
market_ticker = '^NSEI'
risk_free_rate = 0.0677  # Risk-free rate (e.g., 6.77%)
start_date_default = '2020-01-01'
end_date_default = '2024-01-01'

# Streamlit app layout
st.title("NIFTY 50 CAPM & Industry-wise Analysis Dashboard")

# Sidebar inputs
st.sidebar.header("User Input")
selected_sector = st.sidebar.selectbox('Select Sector:', list(set(nifty50_stocks.values())), index=0)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime(start_date_default))
end_date = st.sidebar.date_input("End Date", pd.to_datetime(end_date_default))

# Choose data interval (daily, weekly, monthly)
interval = st.sidebar.selectbox("Select Data Interval:", ['1d', '1wk', '1mo'], index=0)

# Validate dates
if start_date > end_date:
    st.error("End date must be after the start date.")
    st.stop()

# Investment and Risk input
investment_amount = st.sidebar.number_input("Investment Amount (₹):", min_value=1000, value=100000, step=1000)
risk_tolerance = st.sidebar.slider("Risk Tolerance (0-1):", 0.0, 1.0, 0.5)

# Download market data with selected interval
market_data = download_data_with_retry(market_ticker, start_date, end_date, interval)

# Filter stocks by selected sector
stocks = [stock for stock, sector in nifty50_stocks.items() if sector == selected_sector]

# Stock data dictionary
stock_data_dict = {}
for stock in stocks:
    stock_data = download_data_with_retry(stock, start_date, end_date, interval)
    if not stock_data.empty:
        stock_data_dict[stock] = stock_data

# Continue if data is available
if stock_data_dict:
    selected_stock = st.sidebar.selectbox('Select Stock for AO and RSI:', list(stock_data_dict.keys()), index=0)
    stock_data = stock_data_dict[selected_stock]

    # Calculate Awesome Oscillator and RSI
    ao_data = awesome_oscillator(stock_data)
    rsi_data = calculate_rsi(stock_data)

    # Latest AO and RSI values
    latest_ao = ao_data.iloc[-1] if not ao_data.empty else np.nan
    latest_rsi = rsi_data.iloc[-1] if not rsi_data.empty else np.nan

    # Analyze AO and RSI
    ao_signal, ao_trend = analyze_ao(latest_ao)
    rsi_signal, rsi_trend = analyze_rsi(latest_rsi)

    # Display AO and RSI values
    st.subheader(f"Technical Analysis for {selected_stock}")
    st.write(f"**Latest AO (Awesome Oscillator)**: {latest_ao:.2f} ({ao_signal})")
    st.write(f"**Latest RSI (Relative Strength Index)**: {latest_rsi:.2f} ({rsi_signal})")

    # Plot AO and RSI
    st.subheader(f"Awesome Oscillator for {selected_stock}")
    st.line_chart(ao_data)

    st.subheader(f"RSI for {selected_stock}")
    st.line_chart(rsi_data)

    # Portfolio Optimization and CAPM Analysis (using all available stock data)
    stock_data = pd.DataFrame({ticker: stock['Adj Close'] for ticker, stock in stock_data_dict.items()})
    stock_returns = stock_data.pct_change().dropna()
    market_returns = market_data['Adj Close'].pct_change().dropna()

    # Align stock and market data
    stock_returns, market_returns = stock_returns.align(market_returns, join='inner', axis=0)

    if not stock_returns.empty and not market_returns.empty:
        # Optimization setup
        mean_returns = stock_returns.mean() * 252
        cov_matrix = stock_returns.cov() * 252
        num_stocks = len(mean_returns)
        initial_weights = np.array([1.0 / num_stocks] * num_stocks)
        bounds = tuple((0, 1) for _ in range(num_stocks))
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        # Optimize portfolio based on risk tolerance
        optimized_result = minimize(optimize_for_risk, initial_weights, args=(mean_returns, cov_matrix, risk_tolerance, risk_free_rate),
                                    method='SLSQP', bounds=bounds, constraints=constraints)
        optimal_weights = optimized_result.x

        # Display portfolio performance
        st.subheader("Optimal Portfolio Weights")
        portfolio_df = pd.DataFrame({'Stock': mean_returns.index, 'Weight': optimal_weights})
        st.write(portfolio_df)

else:
    st.write("No stock data available for the selected sector.")
