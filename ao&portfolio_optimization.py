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
risk_free_rate = st.sidebar.number_input("Risk-free Rate (annual)", 0.0, 0.2, 0.065, step=0.005, format="%.3f")

# Fetch data
@st.cache_data
def load_price_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    return data.dropna()

df = load_price_data(tickers, start_date, end_date)
returns = df.pct_change().dropna()

# Portfolio optimization (annualised, so the optimizer works with well-scaled numbers)
TRADING_DAYS = 252
mean_returns = returns.mean().values * TRADING_DAYS
cov_matrix = LedoitWolf().fit(returns).covariance_ * TRADING_DAYS
num_assets = len(mean_returns)
bounds = [(0, 1)] * num_assets
equal_weights = np.full(num_assets, 1 / num_assets)
sum_to_one = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

def portfolio_risk(weights):
    return np.sqrt(weights @ cov_matrix @ weights)

def portfolio_return(weights):
    return weights @ mean_returns

def optimize(objective, constraints):
    res = minimize(objective, equal_weights, method='SLSQP', bounds=bounds,
                   constraints=constraints, options={'ftol': 1e-12, 'maxiter': 1000})
    return res.x if res.success else None

def get_efficient_frontier(points=50):
    # Trace the frontier from the minimum-variance portfolio up to the highest-return asset
    min_var_weights = optimize(lambda w: w @ cov_matrix @ w, [sum_to_one])
    targets = np.linspace(portfolio_return(min_var_weights), mean_returns.max(), points)

    results = {"Returns": [], "Risk": [], "Weights": []}
    for target in targets:
        hit_target = {'type': 'eq', 'fun': lambda w, t=target: portfolio_return(w) - t}
        weights = optimize(lambda w: w @ cov_matrix @ w, [sum_to_one, hit_target])
        if weights is not None:
            results["Returns"].append(portfolio_return(weights))
            results["Risk"].append(portfolio_risk(weights))
            results["Weights"].append(weights)

    return pd.DataFrame(results)

def get_max_sharpe_weights():
    return optimize(lambda w: -(portfolio_return(w) - risk_free_rate) / portfolio_risk(w), [sum_to_one])

frontier = get_efficient_frontier()
max_sharpe_weights = get_max_sharpe_weights()

# Plotting
def plot_efficient_frontier(frontier_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sharpe = (frontier_df['Returns'] - risk_free_rate) / frontier_df['Risk']
    scatter = ax.scatter(frontier_df['Risk'], frontier_df['Returns'], c=sharpe, cmap='viridis')
    ax.scatter(portfolio_risk(max_sharpe_weights), portfolio_return(max_sharpe_weights),
               marker='*', s=300, c='red', label='Max Sharpe')
    ax.set_title("Efficient Frontier")
    ax.set_xlabel("Risk (Annual Std. Deviation)")
    ax.set_ylabel("Expected Annual Return")
    ax.legend()
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

# Show the max-Sharpe portfolio (yfinance sorts columns, so label weights from returns.columns)
weights_df = pd.DataFrame({
    "Ticker": returns.columns,
    "Sector": [nifty_50_stocks[t] for t in returns.columns],
    "Weight": max_sharpe_weights.round(4)
}).sort_values(by="Weight", ascending=False)

st.subheader("4️⃣ Suggested Optimal Portfolio (Max Sharpe)")
col1, col2, col3 = st.columns(3)
col1.metric("Expected Annual Return", f"{portfolio_return(max_sharpe_weights):.1%}")
col2.metric("Annual Risk", f"{portfolio_risk(max_sharpe_weights):.1%}")
col3.metric("Sharpe Ratio", f"{(portfolio_return(max_sharpe_weights) - risk_free_rate) / portfolio_risk(max_sharpe_weights):.2f}")
st.dataframe(weights_df)

