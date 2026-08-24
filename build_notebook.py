"""
Generates Technical_Trading_Strategy_Backtester.ipynb using nbformat.
Run this once to (re)build the notebook file.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# ----------------------------------------------------------------------
# Title
# ----------------------------------------------------------------------
md("""\
# Technical Trading Strategy Backtester

A beginner-friendly notebook that builds a simple **Moving Average Crossover**
trading strategy, enhances it with an **RSI filter**, and backtests both against a
**Buy & Hold** benchmark across six markets.

**Markets:** NIFTY 50, S&P 500, NASDAQ Composite, AAPL, RELIANCE, NVIDIA
""")

# ----------------------------------------------------------------------
# 1. Introduction
# ----------------------------------------------------------------------
md("""\
## 1. Introduction

This notebook follows a simple, standard research workflow:

```
Market Data
     |
Trading Rule
     |
Historical Backtest
     |
Performance Analysis
     |
Risk Analysis
     |
Conclusion
```

We keep the strategy intentionally simple:

- **Strategy A — Moving Average Crossover:** 20-day moving average (DMA) vs.
  50-day moving average. `20 DMA > 50 DMA -> Buy`, `20 DMA < 50 DMA -> Sell`.
- **One added indicator — RSI (Relative Strength Index):** used as a simple
  filter on top of Strategy A, so we don't buy when the market already looks
  overbought.
- **Benchmark — Buy & Hold:** hold the asset for the entire period, no trading.
""")

code("""\
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = (12, 6)

print("Libraries imported successfully!")
""")

# ----------------------------------------------------------------------
# 2. Data collection
# ----------------------------------------------------------------------
md("""\
## 2. Market Data

The six markets, mapped to their Yahoo Finance ticker symbols:
""")

code("""\
markets = {
    "NIFTY 50": "^NSEI",
    "S&P 500": "^GSPC",
    "NASDAQ Composite": "^IXIC",
    "AAPL": "AAPL",
    "RELIANCE": "RELIANCE.NS",
    "NVIDIA": "NVDA",
}

PERIOD = "5y"     # last 5 years of history
INTERVAL = "1d"   # daily prices

close_prices = {}
for name, symbol in markets.items():
    print(f"Downloading {name} ({symbol})...")
    data = yf.download(symbol, period=PERIOD, interval=INTERVAL,
                        auto_adjust=True, progress=False)
    # .squeeze() turns the single-column "Close" table into a plain Series
    close_prices[name] = data["Close"].squeeze()

prices = pd.DataFrame(close_prices)
prices = prices.sort_index().ffill().dropna()

print(f"\\nShape: {prices.shape}")
prices.tail()
""")

# ----------------------------------------------------------------------
# 3. Indicators
# ----------------------------------------------------------------------
md("""\
## 3. Indicators

**Moving averages** smooth out day-to-day noise so we can see the underlying
trend. When the short (20-day) average is above the long (50-day) average, the
trend is considered bullish.

**RSI (Relative Strength Index)** measures how strong recent gains have been
compared to recent losses, on a scale of 0-100. A common rule of thumb:
RSI >= 70 means the asset may be **overbought** (risky to buy right now).
""")

code("""\
def compute_rsi(price_series, window=14):
    \"\"\"Classic RSI: average gain vs. average loss over a rolling window.\"\"\"
    delta = price_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    relative_strength = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi
""")

# ----------------------------------------------------------------------
# 4. Trading rule / strategy construction
# ----------------------------------------------------------------------
md("""\
## 4. Trading Rule & Backtest

For every market we build three strategies:

1. **Buy & Hold** — always fully invested.
2. **MA Crossover (20/50)** — invested only while `20 DMA > 50 DMA`.
3. **MA Crossover + RSI Filter** — same as above, but skips a buy signal if
   `RSI >= 70` (overbought) at that time.

To avoid look-ahead bias, a signal generated using **today's** closing price
is only acted on **starting tomorrow** (`signal.shift(1)`), just like a real
trader would.
""")

code("""\
SHORT_WINDOW = 20
LONG_WINDOW = 50
RSI_WINDOW = 14
OVERBOUGHT_LEVEL = 70

strategy_returns = {}   # (market, strategy_name) -> daily return Series
strategy_positions = {} # (market, strategy_name) -> executed position Series (0 or 1)
equity_curves = {}      # market -> DataFrame of growth-of-$1 curves
indicator_data = {}     # market -> DataFrame with price, moving averages, RSI

for name in markets:
    price = prices[name]
    daily_return = price.pct_change().fillna(0)

    short_ma = price.rolling(SHORT_WINDOW).mean()
    long_ma = price.rolling(LONG_WINDOW).mean()
    rsi = compute_rsi(price, RSI_WINDOW)

    ma_signal = (short_ma > long_ma).astype(int)

    rsi_filtered_signal = ma_signal.copy()
    rsi_filtered_signal[(ma_signal == 1) & (rsi >= OVERBOUGHT_LEVEL)] = 0

    strategies = {
        "Buy & Hold": pd.Series(1, index=price.index),
        "MA Crossover (20/50)": ma_signal,
        "MA Crossover + RSI Filter": rsi_filtered_signal,
    }

    curves = {}
    for strat_name, signal in strategies.items():
        executed_position = signal.shift(1).fillna(0)   # trade one day after the signal
        strategy_return = executed_position * daily_return

        strategy_returns[(name, strat_name)] = strategy_return
        strategy_positions[(name, strat_name)] = executed_position
        curves[strat_name] = (1 + strategy_return).cumprod()

    equity_curves[name] = pd.DataFrame(curves)
    indicator_data[name] = pd.DataFrame({
        "Price": price, "20 DMA": short_ma, "50 DMA": long_ma, "RSI": rsi,
    })

print("Backtest complete for all markets and strategies.")
""")

# ----------------------------------------------------------------------
# 5. Equity curves
# ----------------------------------------------------------------------
md("""\
## 5. Equity Curves

The "growth of $1" for each strategy, per market.
""")

code("""\
fig, axes = plt.subplots(3, 2, figsize=(15, 14))
axes = axes.flatten()

for ax, name in zip(axes, markets):
    equity_curves[name].plot(ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")

plt.tight_layout()
plt.show()
""")

# ----------------------------------------------------------------------
# 6. RSI example
# ----------------------------------------------------------------------
md("""\
## 6. RSI Example

A closer look at RSI for one market (AAPL), with the overbought (70) and
oversold (30) reference lines.
""")

code("""\
example_market = "AAPL"
rsi_example = indicator_data[example_market]["RSI"]

plt.figure(figsize=(14, 5))
plt.plot(rsi_example.index, rsi_example, color="purple")
plt.axhline(70, color="red", linestyle="--", label="Overbought (70)")
plt.axhline(30, color="green", linestyle="--", label="Oversold (30)")
plt.title(f"{example_market}: RSI (14-day)")
plt.xlabel("Date")
plt.ylabel("RSI")
plt.legend()
plt.show()
""")

# ----------------------------------------------------------------------
# 7. Performance & risk metrics
# ----------------------------------------------------------------------
md("""\
## 7. Performance Analysis & Risk Analysis

For every market/strategy combination we calculate:

- **Total Return** — overall growth over the full period.
- **Annualized Return (CAGR)** — total return converted to a "per year" rate.
- **Sharpe Ratio** — risk-adjusted return (using a 2% annual risk-free rate).
- **Maximum Drawdown** — worst peak-to-trough decline.
- **Number of Trades** — how many times a position was opened.
- **Win Rate** — the percentage of trades that were profitable.
""")

code("""\
RISK_FREE_RATE = 0.02
TRADING_DAYS_PER_YEAR = 252


def extract_trades(executed_position, price):
    \"\"\"Walk through the position series and return each trade's return.\"\"\"
    trades = []
    in_trade = False
    entry_price = None

    for date, position in executed_position.items():
        if position == 1 and not in_trade:
            in_trade = True
            entry_price = price[date]
        elif position == 0 and in_trade:
            in_trade = False
            exit_price = price[date]
            trades.append((exit_price - entry_price) / entry_price)

    if in_trade:  # still holding at the end of the data: close it out
        trades.append((price.iloc[-1] - entry_price) / entry_price)

    return trades


def compute_metrics(strategy_return, executed_position, price):
    growth = (1 + strategy_return).cumprod()
    total_return = growth.iloc[-1] - 1
    annualized_return = growth.iloc[-1] ** (TRADING_DAYS_PER_YEAR / len(strategy_return)) - 1

    annualized_volatility = strategy_return.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe_ratio = (
        (strategy_return.mean() * TRADING_DAYS_PER_YEAR - RISK_FREE_RATE) / annualized_volatility
        if annualized_volatility > 0 else np.nan
    )

    running_max = growth.cummax()
    max_drawdown = ((growth - running_max) / running_max).min()

    trades = extract_trades(executed_position, price)
    num_trades = len(trades)
    win_rate = (sum(1 for t in trades if t > 0) / num_trades * 100) if num_trades > 0 else np.nan

    return {
        "Total Return (%)": total_return * 100,
        "Annualized Return (%)": annualized_return * 100,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown (%)": max_drawdown * 100,
        "Number of Trades": num_trades,
        "Win Rate (%)": win_rate,
    }
""")

code("""\
results = []
for (market, strategy_name), strategy_return in strategy_returns.items():
    executed_position = strategy_positions[(market, strategy_name)]
    metrics = compute_metrics(strategy_return, executed_position, prices[market])
    metrics["Market"] = market
    metrics["Strategy"] = strategy_name
    results.append(metrics)

results_df = pd.DataFrame(results).set_index(["Market", "Strategy"])
results_df = results_df[[
    "Total Return (%)", "Annualized Return (%)", "Sharpe Ratio",
    "Max Drawdown (%)", "Number of Trades", "Win Rate (%)",
]].round(2)

results_df
""")

# ----------------------------------------------------------------------
# 8. Strategy comparison charts
# ----------------------------------------------------------------------
md("""\
## 8. Strategy Comparison

Total return and Sharpe ratio, side by side across all markets and strategies.
""")

code("""\
total_return_pivot = results_df["Total Return (%)"].unstack("Strategy")
total_return_pivot.plot(kind="bar", figsize=(14, 6))
plt.title("Total Return (%) by Market and Strategy")
plt.ylabel("Total Return (%)")
plt.xticks(rotation=0)
plt.legend(title="Strategy")
plt.show()
""")

code("""\
sharpe_pivot = results_df["Sharpe Ratio"].unstack("Strategy")
sharpe_pivot.plot(kind="bar", figsize=(14, 6))
plt.title("Sharpe Ratio by Market and Strategy")
plt.ylabel("Sharpe Ratio")
plt.xticks(rotation=0)
plt.legend(title="Strategy")
plt.show()
""")

# ----------------------------------------------------------------------
# 9. Risk analysis: drawdowns
# ----------------------------------------------------------------------
md("""\
## 9. Risk Analysis: Drawdowns

Comparing the drawdown profile of Buy & Hold against the RSI-filtered
crossover strategy for every market.
""")

code("""\
fig, axes = plt.subplots(3, 2, figsize=(15, 14))
axes = axes.flatten()

for ax, name in zip(axes, markets):
    for strat_name in ["Buy & Hold", "MA Crossover + RSI Filter"]:
        growth = equity_curves[name][strat_name]
        running_max = growth.cummax()
        drawdown = (growth - running_max) / running_max * 100
        ax.plot(drawdown.index, drawdown, label=strat_name)

    ax.set_title(name)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()
""")

# ----------------------------------------------------------------------
# 10. Conclusion
# ----------------------------------------------------------------------
md("""\
## 10. Conclusion

Fill in your own observations after running the notebook. Some questions to
think about:

- Did the MA Crossover strategy beat Buy & Hold on every market, or only
  some? Which ones?
- Did adding the RSI filter improve the Sharpe ratio, reduce the max
  drawdown, or both — and did it cost any total return?
- Which market had the highest win rate? Which had the most trades?
- Individual stocks (AAPL, RELIANCE, NVIDIA) vs. broad indices (NIFTY 50,
  S&P 500, NASDAQ Composite) — did the strategy behave differently on
  single stocks compared to diversified indices?

### Next steps

- Try different moving-average windows (e.g. 10/30, 50/200) in Section 4.
- Try a different RSI threshold, or use RSI to also filter *sell* signals.
- Add transaction costs (a small % deducted per trade) to see how they
  affect the results.
""")

nb["cells"] = cells

with open("Technical_Trading_Strategy_Backtester.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook written to Technical_Trading_Strategy_Backtester.ipynb")
