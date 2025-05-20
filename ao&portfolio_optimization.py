import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go
from scipy.optimize import minimize
import time

# ------------------ Caching Stock Data with Retry ------------------
@st.cache_data
def download_data_with_retry(ticker, start_date, end_date, interval='1d', retries=5, delay=5):
    for i in range(retries):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
            if not data.empty:
                return data
        except Exception:
            time.sleep(delay)
    return pd.DataFrame()

# ------------------ Technical Indicators ------------------
def awesome_oscillator(stock_data):
    median_price = (stock_data['High'] + stock_data['Low']) / 2
    ao = median_price.rolling(window=5).mean() - median_price.rolling(window=34).mean()
    return ao

def calculate_rsi(stock_data, periods=14):
    delta = stock_data['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def analyze_rsi(rsi_value):
    if rsi_value > 70:
        return 'Overbought (Sell signal)', 'Bearish'
    elif rsi_value < 30:
        return 'Oversold (Buy signal)', 'Bullish'
    else:
        return 'Neutral', 'Neutral'

def analyze_ao(ao_value):
    if ao_value.iloc[-1] > 0:
        return 'Bullish', 'Buy signal'
    elif ao_value.iloc[-1] < 0:
        return 'Bearish', 'Sell signal'
    else:
        return 'Neutral', 'Hold'

# ------------------ Beta Calculation ------------------
def calculate_beta(stock_returns, market_returns):
    covariance_matrix = np.cov(stock_returns, market_returns)
    beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
    return beta

# ------------------ App Layout ------------------
st.title("📊 NIFTY 50 Technical & Portfolio Optimization Dashboard")

# ------------------ Sidebar Inputs ------------------
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

market_ticker = '^NSEI'
risk_free_rate = 0.0677
start_date_default = '2020-01-01'
end_date_default = '2024-01-01'

# Sidebar Inputs
selected_sector = st.sidebar.selectbox('Select Sector:', sorted(set(nifty50_stocks.values())))
start_date = st.sidebar.date_input("Start Date", pd.to_datetime(start_date_default))
end_date = st.sidebar.date_input("End Date", pd.to_datetime(end_date_default))
interval = st.sidebar.selectbox("Data Interval:", ['1d', '1wk', '1mo'], index=0)
investment_amount = st.sidebar.number_input("Investment Amount (₹):", min_value=1000, value=100000, step=1000)
risk_tolerance = st.sidebar.slider("Risk Tolerance (0-Low ➡️ 1-High):", 0.0, 1.0, 0.5)

if start_date > end_date:
    st.error("❌ End date must be after start date.")
    st.stop()

# ------------------ Market Data ------------------
market_data = download_data_with_retry(market_ticker, start_date, end_date, interval)

# ------------------ Sector-wise Stock Data ------------------
sector_stocks = [s for s, sector in nifty50_stocks.items() if sector == selected_sector]
stock_data_dict = {}
for stock in sector_stocks:
    data = download_data_with_retry(stock, start_date, end_date, interval)
    if not data.empty:
        stock_data_dict[stock] = data

if not stock_data_dict:
    st.warning("No stock data available for selected sector.")
    st.stop()

# ------------------ Technical Analysis ------------------
selected_stock = st.sidebar.selectbox('Select Stock for Technical Indicators:', list(stock_data_dict.keys()))
stock_data = stock_data_dict[selected_stock]
ao_data = awesome_oscillator(stock_data)
rsi_data = calculate_rsi(stock_data)
latest_ao = ao_data.iloc[-1]
latest_rsi = rsi_data.iloc[-1]
ao_signal, ao_trend = analyze_ao(pd.Series([latest_ao]))
rsi_signal, rsi_trend = analyze_rsi(latest_rsi)

st.subheader(f"📈 Technical Analysis: {selected_stock}")
st.write(f"**Awesome Oscillator (AO):** {latest_ao:.2f} → {ao_signal}")
st.line_chart(ao_data)
st.write(f"**Relative Strength Index (RSI):** {latest_rsi:.2f} → {rsi_signal}")
st.line_chart(rsi_data)

# ------------------ Portfolio Analysis ------------------
adj_close_df = pd.DataFrame({ticker: data['Adj Close'] for ticker, data in stock_data_dict.items()})
stock_returns = adj_close_df.pct_change().dropna()
market_returns = market_data['Adj Close'].pct_change().dropna()
stock_returns, market_returns = stock_returns.align(market_returns, join='inner', axis=0)

if stock_returns.empty or market_returns.empty:
    st.warning("No sufficient return data.")
    st.stop()

betas = {stock: calculate_beta(stock_returns[stock], market_returns) for stock in stock_returns.columns}
st.subheader("📌 CAPM: Stock Betas")
st.dataframe(pd.DataFrame(betas.items(), columns=['Stock', 'Beta']))

# Portfolio optimization
def portfolio_statistics(weights, mean_returns, cov_matrix, risk_free_rate):
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (port_return - risk_free_rate) / port_vol
    return port_return, port_vol, sharpe

def optimize_for_risk(weights, mean_returns, cov_matrix, risk_tolerance, risk_free_rate):
    ret, vol, _ = portfolio_statistics(weights, mean_returns, cov_matrix, risk_free_rate)
    return (1 - risk_tolerance) * vol - risk_tolerance * ret

mean_returns = stock_returns.mean() * 252
cov_matrix = stock_returns.cov() * 252
num_stocks = len(mean_returns)
init_weights = np.ones(num_stocks) / num_stocks
bounds = tuple((0, 1) for _ in range(num_stocks))
constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

result = minimize(optimize_for_risk, init_weights,
                  args=(mean_returns, cov_matrix, risk_tolerance, risk_free_rate),
                  method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = result.x
port_return, port_volatility, sharpe_ratio = portfolio_statistics(optimal_weights, mean_returns, cov_matrix, risk_free_rate)

st.subheader("📊 Optimal Portfolio Allocation")
st.dataframe(pd.DataFrame({'Stock': mean_returns.index, 'Weight': optimal_weights}))

st.subheader("📈 Portfolio Performance Metrics")
st.write(f"**Expected Annual Return:** {port_return:.2%}")
st.write(f"**Annual Volatility:** {port_volatility:.2%}")
st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")

# Efficient Frontier
st.subheader("🧭 Efficient Frontier")
target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 100)
frontier_vol = []
for target in target_returns:
    ef_constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                      {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - target}]
    result = minimize(lambda x: np.sqrt(np.dot(x.T @ cov_matrix, x)), init_weights,
                      method='SLSQP', bounds=bounds, constraints=ef_constraints)
    frontier_vol.append(result['fun'])

fig = go.Figure()
fig.add_trace(go.Scatter(x=frontier_vol, y=target_returns, mode='lines', name='Efficient Frontier'))
st.plotly_chart(fig)

# Recommendation
if sharpe_ratio > risk_tolerance:
    st.success("✅ Portfolio has a favorable risk-adjusted return. Suitable for investment.")
else:
    st.warning("⚠️ Portfolio may not align with your risk tolerance. Consider adjusting allocations.")
