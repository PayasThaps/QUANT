
# Portfolio Optimization and Trend Analysis Dashboard

This Streamlit app allows users to explore NIFTY 50 stocks by sector, optimize portfolios based on the Capital Asset Pricing Model (CAPM), analyze risk-adjusted returns, and track trends using the Awesome Oscillator.

## Features

- **Sector-wise Stock Selection**: Choose from various sectors of NIFTY 50 companies to analyze and optimize.
- **Portfolio Optimization**: Maximize the Sharpe ratio using Markowitz's portfolio theory.
- **Efficient Frontier**: Visualize the efficient frontier for optimal portfolios.
- **Risk and Investment Customization**: Specify your risk tolerance and the amount you want to invest.
- **Awesome Oscillator**: Track stock price trends and visualize buy/sell signals.
- **Buy/Sell Suggestions**: Get suggestions based on your risk tolerance and portfolio performance.

## Installation

1. **Clone this repository** (or download the script):
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   ```

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   If you don't have a `requirements.txt` file, you can manually install the dependencies:
   ```bash
   pip install streamlit yfinance statsmodels plotly scipy
   ```

3. **Run the Streamlit App**:
   ```bash
   streamlit run ao&portfolio_optimization.py
   ```

4. Open your browser and go to `http://localhost:8501` (or the URL provided by the terminal).

## Usage

1. **Select Sector**: Use the sidebar to select the sector you want to analyze.
2. **Specify Date Range**: Choose the start and end dates for historical data analysis.
3. **Customize Investment & Risk**: Enter your desired investment amount and adjust the risk tolerance slider to match your risk profile.
4. **Visualize**:
   - **Optimal Portfolio Weights**: View the recommended allocation of your investment across stocks.
   - **Efficient Frontier**: Analyze the risk-return tradeoff for various portfolios.
   - **Awesome Oscillator**: See the stock trends and potential buy/sell signals.

## Example

Here's a basic example of how to use this app:

1. Select **Financials** as the sector.
2. Set the investment amount to **₹100,000** and a risk tolerance of **0.6**.
3. Choose a date range of **2020-01-01** to **2024-01-01**.
4. View the optimized portfolio and trend analysis for the selected stocks.

## Theory & Formulas

### 1. **Capital Asset Pricing Model (CAPM)**

The CAPM formula is used to calculate the expected return of a stock based on its risk compared to the overall market:

\[
    E(R_i) = R_f + \beta_i \cdot (E(R_m) - R_f)
\]

Where:
- \(E(R_i)\) = Expected return of the stock
- \(R_f\) = Risk-free rate
- \(\beta_i\) = Stock’s sensitivity to market movements (Beta)
- \(E(R_m)\) = Expected return of the market

### 2. **Sharpe Ratio**

The Sharpe Ratio measures the performance of an investment compared to a risk-free asset, after adjusting for its risk. It is defined as:

\[
    Sharpe Ratio = \frac{E(R_p) - R_f}{\sigma_p}
\]

Where:
- \(E(R_p)\) = Expected return of the portfolio
- \(R_f\) = Risk-free rate
- \(\sigma_p\) = Standard deviation (volatility) of the portfolio’s excess return

### 3. **Efficient Frontier**

The efficient frontier represents the set of optimal portfolios that offer the highest expected return for a given level of risk. The portfolios on the efficient frontier are calculated using:

\[
    Minimize(\sigma_p^2) = w^T \Sigma w
\]

Subject to:
\[
    \sum w_i = 1
\]
and
\[
    E(R_p) = \sum w_i \cdot E(R_i)
\]

Where:
- \(\Sigma\) = Covariance matrix of stock returns
- \(w\) = Portfolio weights
- \(E(R_p)\) = Expected return of the portfolio

### 4. **Awesome Oscillator (AO)**

The Awesome Oscillator is a technical indicator used to gauge market momentum. It is calculated as the difference between the 5-period and 34-period simple moving averages of the median price (high + low / 2):

\[
    AO = SMA(5) - SMA(34)
\]

Where:
- \(SMA(n)\) = Simple moving average over \(n\) periods
- Median Price = \( \frac{High + Low}{2} \)

## Requirements

- Python 3.7 or higher
- Streamlit
- yfinance
- statsmodels
- plotly
- scipy
- pyngrok (for running on Colab or other remote servers)

## Running on Google Colab

To run the app in Google Colab, use the following commands to install the necessary packages and set up `ngrok` to create a tunnel for the Streamlit app:

```bash
!pip install streamlit pyngrok
```

Then, add this code in your Colab cell to run the app:

```python
from pyngrok import ngrok
import os

public_url = ngrok.connect(port=8501)
print("Streamlit App URL:", public_url)

os.system("streamlit run ao&portfolio_optimization.py &")
```

Access your app via the public URL generated by `ngrok`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
