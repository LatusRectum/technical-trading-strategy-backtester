# Technical Trading Strategy Backtester

Jupyter notebook that I build for a simple **Moving Average Crossover**
trading strategy, enhances it with an **RSI filter**, and backtests both against a
**Buy & Hold** benchmark, using real historical data from
[`yfinance`](https://pypi.org/project/yfinance/).

**Markets:** NIFTY 50 (`^NSEI`), S&P 500 (`^GSPC`), NASDAQ Composite (`^IXIC`),
AAPL, RELIANCE (`RELIANCE.NS`), NVIDIA (`NVDA`)

## Workflow

```
Market Data -> Trading Rule -> Historical Backtest -> Performance Analysis -> Risk Analysis -> Conclusion
```

## Strategies

- **Buy & Hold** — always fully invested (the benchmark).
- **Strategy A — MA Crossover (20/50)** — invested only while the 20-day moving
  average is above the 50-day moving average.
- **MA Crossover + RSI Filter** — same crossover rule, but skips a buy signal
  when RSI (14-day) is >= 70 (overbought).

## Metrics calculated

- Total Return
- Annualized Return (CAGR)
- Sharpe Ratio
- Maximum Drawdown
- Number of Trades
- Win Rate

## What's inside

The notebook (`Technical_Trading_Strategy_Backtester.ipynb`) will comprise of 10 sections:

1. Introduction
2. Market data
3. Indicators (moving averages, RSI)
4. Trading rule & backtest
5. Equity curves
6. RSI example
7. Performance & risk metrics
8. Strategy comparison charts
9. Risk analysis: drawdowns
10. Conclusion

All the code is written to be simple and readable, with a comment or markdown
explanation above every step.

## Getting started

### 1. Install the required packages

```bash
pip install -r requirements.txt
```

```bash
jupyter notebook
```

Then open `Technical_Trading_Strategy_Backtester.ipynb` and run all cells
(`Cell > Run All`).

> **Note:** This notebook downloads live data from Yahoo Finance each time you run it. Results will look different
> every time you re-run it, since the market keeps moving.

## Project structure

```
technical-trading-strategy-backtester/
├── Technical_Trading_Strategy_Backtester.ipynb   # the main notebook
├── build_notebook.py                             # script that generates the notebook
├── requirements.txt                              # Python dependencies
├── README.md
└── .gitignore
```

## Customizing it

- Change `SHORT_WINDOW` / `LONG_WINDOW` in Section 4 to try different moving-average
  pairs (e.g. 10/30, 50/200).
- Change `OVERBOUGHT_LEVEL` to try a different RSI threshold.
- Add more markets by extending the `markets` dictionary in Section 2.
