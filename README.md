# Equity Pairs Trading Infrastructure

![CI](https://github.com/Minhal12345/equity-pairs-pipeline/actions/workflows/ci.yml/badge.svg)

A Python pipeline for statistical arbitrage — ingests end-of-day equity data, tests for cointegration between stock pairs, generates z-score trading signals, and backtests the strategy. Built with a medallion architecture and automated data quality monitoring.

## Architecture

```
data/raw/           →    data/processed/        →    data/output/
(per-ticker OHLCV)       (reconciliation,            (backtest results,
                          cointegration,              charts, quality
                          signals)                    reports)
```

```
ingest.py → reconcile.py → cointegration.py → signals.py → backtest.py → report.py
                │
                └→ monitor.py (7 data quality checks → quality_report.json)
```

**Modules:**

- `src/ingest.py` — Dual-source ingestion (yfinance primary, Alpha Vantage secondary), per-ticker Parquet output
- `src/reconcile.py` — Cross-source alignment, divergence flagging, missing-day detection
- `src/cointegration.py` — Engle-Granger test on all within-sector pairs, OLS hedge ratios, OU half-life, Benjamini-Hochberg FDR correction
- `src/signals.py` — Spread z-score computation with entry/exit/stop-loss state machine
- `src/backtest.py` — Strategy simulation, Sharpe/drawdown/win rate, equal-weight portfolio aggregation
- `src/oos_validation.py` — Out-of-sample validation with train/test split
- `src/monitor.py` — 7 automated data quality checks (staleness, coverage, schema, outliers, zero volume, divergence drift, price sanity)

## Methodology

**Cointegration (Engle-Granger):** For each within-sector pair, regress price A on price B via OLS to get the hedge ratio β, then test whether the residuals are stationary using an augmented Dickey-Fuller test. After testing all 54 pairs, apply Benjamini-Hochberg FDR correction to control false discovery rate.

**Signal generation:** Compute the spread as `price_A - β · price_B`, then standardize with a 60-day rolling z-score. Enter long at z < −2, enter short at z > +2, exit on z-score sign reversal, stop-loss at |z| > 3.

**Backtesting:** Simulate daily P&L as `position × spread_change`, compute annualized Sharpe (√252), max drawdown, and win rate. Aggregate across pairs into an equal-weight portfolio.

**Out-of-sample validation:** Split at October 2023 — 2 years training, 1 year test. Re-run cointegration on training data only, apply training-period parameters to the test period without re-estimation.

## Results

### Cointegration Heatmap

![Cointegration P-Values](data/output/charts/cointegration_heatmap.png)

*Engle-Granger p-values for all within-sector pairs. Green cells indicate raw significance (p < 0.05). After BH-FDR correction, no pairs remain significant.*

### Spread Z-Score — Top Pairs

![Spread Z-Score PEP/YUM](data/output/charts/spread_zscore_PEP_YUM.png)

*PEP/YUM spread z-score (Sharpe 1.60) with entry (±2) and stop-loss (±3) thresholds.*

![Spread Z-Score V/MA](data/output/charts/spread_zscore_V_MA.png)

*V/MA spread z-score (Sharpe 1.23).*

### P&L Curves

![P&L PEP/YUM](data/output/charts/pnl_curve_PEP_YUM.png)

*PEP/YUM cumulative P&L — 203% total return.*

![Portfolio P&L](data/output/charts/portfolio_pnl.png)

*Equal-weight portfolio across 9 raw-significant pairs.*

### Backtest Summary

| Pair | Total Return % | Sharpe | Max DD % | Trades | Win Rate % |
|------|---------------|--------|----------|--------|------------|
| PEP/YUM | 203.1 | 1.60 | -17.3 | 19 | 78.9 |
| PEP/COST | 65.5 | 1.27 | -16.4 | 17 | 76.5 |
| V/MA | 358.2 | 1.23 | -35.2 | 17 | 82.4 |
| PEP/WMT | 56.9 | 1.18 | -17.5 | 15 | 66.7 |
| PEP/MCD | 113.5 | 1.15 | -24.6 | 19 | 68.4 |
| MSFT/META | 79.6 | 0.66 | -36.8 | 14 | 71.4 |
| JNJ/MRK | 23.4 | 0.62 | -11.5 | 16 | 62.5 |
| GOOGL/AMZN | -49.1 | 0.60 | -88.3 | 16 | 62.5 |
| JNJ/PFE | 16.3 | 0.46 | -13.1 | 14 | 57.1 |
| **Portfolio (EW)** | **96.4** | **1.51** | **-14.8** | **147** | **69.6** |

9 of 54 pairs showed raw significance at p < 0.05. After FDR correction, 0 survived. Out-of-sample Sharpe degraded from 1.44 to 0.15 — the in-sample edge does not hold up under standard statistical rigor.

## Data

| Source | Role | Coverage |
|--------|------|----------|
| yfinance | Primary | 30 tickers × 3 years daily OHLCV |
| Alpha Vantage | Secondary (reconciliation) | Same tickers, free API |

**Universe:** 30 S&P 500 constituents across 7 sectors:

| Sector | Tickers | Pairs |
|--------|---------|-------|
| Tech | AAPL, MSFT, GOOGL, AMZN, NVDA, META | 15 |
| Consumer | KO, PEP, MCD, YUM, COST, WMT | 15 |
| Financials | JPM, BAC, GS, MS | 6 |
| Energy | XOM, CVX, COP, SLB | 6 |
| Healthcare | JNJ, PFE, MRK, ABBV | 6 |
| Payments | V, MA, PYPL | 3 |
| Retail | HD, LOW, NKE | 3 |

**Window:** October 2021 – September 2024 (753 NYSE trading days).

## Usage

```bash
git clone https://github.com/Minhal12345/equity-pairs-pipeline.git
cd equity-pairs-pipeline
pip install -r requirements.txt

# Run full pipeline
python main.py --skip-av

# Run with Alpha Vantage reconciliation
export ALPHAVANTAGE_API_KEY=your_key
python main.py --av-key $ALPHAVANTAGE_API_KEY

# Out-of-sample validation
python src/oos_validation.py

# Tests
pytest tests/ -v
```

## Tech Stack

Python, pandas, NumPy, statsmodels, matplotlib, Apache Parquet, pytest, GitHub Actions CI.
