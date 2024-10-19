import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go
from scipy.optimize import minimize
import requests
import time

# Function to download stock data with retries and caching using Streamlit's cache
@st.cache_data
def download_data_with_retry(ticker, start_date, end_date, retries=5, delay=5):
    for i in range(retries):
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if not data.empty:
                return data
        except Exception as e:
            time.sleep(delay)
    return pd.DataFrame()  # Return empty DataFrame if all retries fail

# List of NIFTY 50 companies and their sectors
nifty50_stocks = {
    'RELIANCE.NS': 'Energy', 'TCS.NS': 'IT', 'HDFCBANK.NS': 'Financials', 'INFY.NS': 'IT',
    'ICICIBANK.NS': 'Financials', 'HINDUNILVR.NS': 'Consumer Goods', 'ITC.NS': 'Consumer Goods',
    'KOTAKBANK.NS': 'Financials', 'SBIN.NS': 'Financials', 'LT.NS': 'Infrastructure',
    'BHARTIARTL.NS': 'Telecom', 'AXISBANK.NS': 'Financials', 'HDFC.NS': 'Financials',
    'BAJFINANCE.NS': 'Financials', 'ASIANPAINT.NS': 'Consumer Goods', 'MARUTI.NS': 'Automobile',
    'HCLTECH.NS': 'IT', 'WIPRO.NS': 'IT', 'ULTRACEMCO.NS': 'Cement', 'ONGC.NS': 'Energy',
    'TITAN.NS': 'Consumer Goods', 'BAJAJFINSV.NS': 'Financials', 'SUNPHARMA.NS': 'Pharmaceuticals',
    'NTPC.NS': 'Energy', 'NESTLEIND.NS': 'Consumer Goods', 'POWERGRID.NS': 'Energy',
    'TATAMOTORS.NS': 'Automobile', 'M&M.NS': 'Automobile', 'ADANIENT.NS': 'Conglomerate',
    'INDUSINDBK.NS': 'Financials', 'HINDALCO.NS': 'Metals', 'JSWSTEEL.NS': 'Metals',
    'DIVISLAB.NS': 'Pharmaceuticals', 'DRREDDY.NS': 'Pharmaceuticals', 'GRASIM.NS': 'Cement',
    'BPCL.NS': 'Energy', 'BRITANNIA.NS': 'Consumer Goods', 'CIPLA.NS': 'Pharmaceuticals',
    'HEROMOTOCO.NS': 'Automobile', 'COALINDIA.NS': 'Metals', 'EICHERMOT.NS': 'Automobile',
    'APOLLOHOSP.NS': 'Healthcare', 'TATACONSUM.NS': 'Consumer Goods', 'SBILIFE.NS': 'Insurance',
    'ICICIGI.NS': 'Insurance', 'HDFCLIFE.NS': 'Insurance', 'TECHM.NS': 'IT', 'BAJAJ-AUTO.NS': 'Automobile',
    'DABUR.NS': 'Consumer Goods'
}

# OpenAI API function for ChatGPT integration
def get_gpt_analysis(api_key, prompt):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }
    json_data = {
        'model': 'gpt-3.5-turbo',
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.7,
    }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=json_data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code}, {response.text}"

# Streamlit app layout
st.title("NIFTY 50 CAPM & Industry-wise Analysis Dashboard with ChatGPT Integration")

# Sidebar inputs
st.sidebar.header("User Input")
selected_sector = st.sidebar.selectbox('Select Sector:', list(set(nifty50_stocks.values())), index=0)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input("End Date", pd.to_datetime('2024-01-01'))

# Validate dates
if start_date > end_date:
    st.error("End date must be after the start date.")
    st.stop()

# OpenAI API key (directly integrated based on your request)
api_key = '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'

# Investment and Risk input
investment_amount = st.sidebar.number_input("Investment Amount (₹):", min_value=1000, value=100000, step=1000)
risk_tolerance = st.sidebar.slider("Risk Tolerance (0-1):", 0.0, 1.0, 0.5)

# Download market data
market_data = download_data_with_retry('^NSEI', start_date, end_date)

# Filter stocks by selected sector
stocks = [stock for stock, sector in nifty50_stocks.items() if sector == selected_sector]

stock_data_dict = {}
for stock in stocks:
    stock_data = download_data_with_retry(stock, start_date, end_date)
    if not stock_data.empty:
        stock_data_dict[stock] = stock_data

# Continue if data is available
if stock_data_dict:
    stock_data = pd.DataFrame({ticker: stock['Adj Close'] for ticker, stock in stock_data_dict.items()})
    stock_returns = stock_data.pct_change().dropna()
    market_returns = market_data['Adj Close'].pct_change().dropna()

    # Align stock and market data
    stock_returns, market_returns = stock_returns.align(market_returns, join='inner', axis=0)

    if not stock_returns.empty and not market_returns.empty:
        # Vectorized portfolio statistics calculation
        def portfolio_statistics(weights, mean_returns, cov_matrix, risk_free_rate):
            portfolio_return = np.dot(weights, mean_returns)  # Vectorized for return
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # Vectorized volatility
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            return portfolio_return, portfolio_volatility, sharpe_ratio

        def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
            return -portfolio_statistics(weights, mean_returns, cov_matrix, risk_free_rate)[2]

        # Optimization setup
        mean_returns = stock_returns.mean() * 252
        cov_matrix = stock_returns.cov() * 252
        num_stocks = len(mean_returns)
        initial_weights = np.array([1.0 / num_stocks] * num_stocks)
        bounds = tuple((0, 1) for _ in range(num_stocks))
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        # Optimize portfolio
        optimized_result = minimize(negative_sharpe_ratio, initial_weights, args=(mean_returns, cov_matrix, 0.0677),
                                    method='SLSQP', bounds=bounds, constraints=constraints)
        optimal_weights = optimized_result.x

        # Display portfolio performance
        st.subheader("Optimal Portfolio Weights")
        portfolio_df = pd.DataFrame({'Stock': mean_returns.index, 'Weight': optimal_weights})
        st.write(portfolio_df)

        # Generate the prompt for ChatGPT analysis
        prompt = f"Analyze the portfolio with these weights: {portfolio_df.to_string()}.\n" \
                 f"Consider the current market data and suggest improvements based on {investment_amount} investment " \
                 f"with a risk tolerance of {risk_tolerance}. Also, provide insights on market trends in the {selected_sector} sector."

        # Get analysis from ChatGPT
        analysis = get_gpt_analysis(api_key, prompt)
        st.subheader("GPT-3 Analysis and Suggestions")
        st.write(analysis)

else:
    st.write("No stock data available for the selected sector.")

