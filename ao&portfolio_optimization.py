import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize

# Display settings
st.set_page_config(page_title="NIFTY 50 Portfolio Optimizer", layout="wide")
st.title("📊 NIFTY 50 Portfolio Optimization Dashboard")

# Load NIFTY 50 stocks
nifty_50_stocks = {
    'RELIANCE.NS': 'Energy', 'TCS.NS': 'IT', 'HDFCBANK.NS': 'Banking',
    'INFY.NS': 'IT', 'HINDUNILVR.NS': 'FMCG', 'ICICIBANK.NS': 'Banking',
    'SBIN.NS': 'Banking', 'KOTAKBANK.NS': 'Banking', 'BAJFINANCE.NS': 'NBFC',
    'ITC.NS': 'FMCG'
}

tickers = list(nifty_50_stocks.keys())
sectors = list(nifty_50_stocks.values())

# Sidebar controls
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))

# Fetch data
@st.cache_data
def load_price_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)["Adj Close"]
    return data.dropna()

df = load_price_data(tickers, start_date, end_date)
returns = df.pct_change().dropna()

# Portfolio optimization
def calculate_portfolio_variance(weights, cov_matrix):
    return weights.T @ cov_matrix @ weights

def get_efficient_frontier(returns, points=50):
    mean_returns = returns.mean()
    cov_matrix = LedoitWolf().fit(returns).covariance_
    num_assets = len(mean_returns)

    results = {"Returns": [], "Risk": [], "Weights": []}
    bounds = [(0, 1)] * num_assets
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    for _ in range(points):
        guess = np.random.dirichlet(np.ones(num_assets), size=1)[0]
        res = minimize(
            calculate_portfolio_variance, guess,
            args=(cov_matrix,), method='SLSQP',
            bounds=bounds, constraints=constraints
        )
        if res.success:
            port_variance = res.fun
            port_return = np.sum(res.x * mean_returns)
            results["Returns"].append(port_return)
            results["Risk"].append(np.sqrt(port_variance))
            results["Weights"].append(res.x)

    return pd.DataFrame(results)

frontier = get_efficient_frontier(returns)

# Plotting
def plot_efficient_frontier(frontier_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(frontier_df['Risk'], frontier_df['Returns'], c=frontier_df['Returns'] / frontier_df['Risk'], cmap='viridis')
    ax.set_title("Efficient Frontier")
    ax.set_xlabel("Risk (Std. Deviation)")
    ax.set_ylabel("Expected Return")
    fig.colorbar(scatter, label="Sharpe Ratio")
    st.pyplot(fig)

# Show correlation heatmap
def plot_correlation_matrix(returns_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(returns_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Show price trends
def plot_price_trends(df):
    st.line_chart(df)

# Render sections
st.subheader("1️⃣ Price Trends")
plot_price_trends(df)

st.subheader("2️⃣ Correlation Matrix")
plot_correlation_matrix(returns)

st.subheader("3️⃣ Efficient Frontier")
plot_efficient_frontier(frontier)

# Show top optimal portfolio
top_portfolio = frontier.iloc[frontier['Returns'].idxmax()]
weights_df = pd.DataFrame({
    "Ticker": tickers,
    "Sector": sectors,
    "Weight": top_portfolio["Weights"]
}).sort_values(by="Weight", ascending=False)

st.subheader("4️⃣ Suggested Optimal Portfolio")
st.dataframe(weights_df)

