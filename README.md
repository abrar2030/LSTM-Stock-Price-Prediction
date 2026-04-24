# LSTM Stock Price Prediction

A complete end-to-end LSTM pipeline for stock price forecasting in Python. Covers data download, feature engineering, model training, evaluation, and benchmark comparison with honest discussion of where the model fails.

## What This Notebook Covers

| Section                 | Description                                     |
| ----------------------- | ----------------------------------------------- |
| 1. Libraries            | Install and import all dependencies             |
| 2. Data Download        | SPY price data via yfinance (2018 to 2024)      |
| 3. Feature Engineering  | 15 technical and statistical features           |
| 4. Train/Val/Test Split | Temporal 70/15/15 split with leak prevention    |
| 5. LSTM Architecture    | 3-layer stacked LSTM with dropout               |
| 6. Training             | EarlyStopping and ReduceLROnPlateau callbacks   |
| 7. Evaluation           | RMSE, MAE, MAPE, R-squared on held-out test set |
| 8. Benchmark Comparison | Naive random walk baseline                      |
| 9. Directional Accuracy | Up/down prediction analysis                     |
| 10. Key Takeaways       | Honest assessment of model limits               |

---

## Features Engineered

| Group      | Features                                       |
| ---------- | ---------------------------------------------- |
| Price      | Log return, High-low spread, Open-close return |
| Trend      | SMA 5/20/60 (deviation), EMA 12/26, MACD       |
| Volatility | Realised volatility at 5/20/60-day windows     |
| Volume     | 30-day rolling z-score normalised volume       |
| Momentum   | RSI normalised, lagged returns at 1/5/22 days  |

---

## Model Architecture

```
Input (60 days x 15 features)
    -> LSTM (128 units, return_sequences=True)
    -> Dropout (0.2)
    -> LSTM (64 units, return_sequences=True)
    -> Dropout (0.2)
    -> LSTM (32 units, return_sequences=False)
    -> Dropout (0.2)
    -> Dense (16, ReLU)
    -> Dropout (0.1)
    -> Dense (1, Linear)
```

Total parameters: ~170,000

---

## Results (SPY, Test Set 2023 to 2024)

| Metric               | LSTM (Ours)         | Naive Baseline      |
| -------------------- | ------------------- | ------------------- |
| RMSE ($)             | see notebook output | see notebook output |
| MAE ($)              | see notebook output | see notebook output |
| MAPE (%)             | see notebook output | see notebook output |
| R-squared            | see notebook output | see notebook output |
| Directional Accuracy | 38.7%               | ~50%                |

Note: The directional accuracy of 38.7% is below random guessing (50%). This is an expected and honest result. LSTMs that fit price levels well do not necessarily capture directional movement. Profitable trading requires directional accuracy above 55% sustained out-of-sample, not just low RMSE.

---

## Known Issue: yfinance MultiIndex Columns

Recent versions of yfinance return a MultiIndex column structure. If your feature names appear as `Log_Return-`, `HL_Spread-` (with a trailing dash) and the correlation matrix is blank, add the following fix immediately after the `yf.download()` call:

```python
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)
raw.columns.name = None
```

---

## Requirements

```
yfinance>=0.2.60
tensorflow>=2.14.0
scikit-learn>=1.3.0
arch>=6.0.0
matplotlib>=3.7.0
seaborn>=0.13.0
pandas>=2.0.0
numpy>=1.24.0
```

Install all at once:

```bash
pip install yfinance tensorflow scikit-learn arch matplotlib seaborn pandas numpy
```

---

## Quick Start

```bash
git clone https://github.com/abrar2030/LSTM-Stock-Price-Prediction
cd LSTM-Stock-Price-Prediction
jupyter notebook LSTM-Stock-Price-Prediction.ipynb
```

To change the ticker, edit the CONFIG block at the top of Cell 2:

```python
TICKER     = 'SPY'          # change to AAPL, MSFT, GLD, etc.
START_DATE = '2018-01-01'
END_DATE   = '2024-12-31'
LOOKBACK   = 60
```

---

## Limitations

- Operates on daily data only. Intraday flash crashes are not captured.
- Cannot predict black swan events with no historical analog in training data.
- Price-level prediction does not equal a profitable trading strategy.
- Walk-forward retraining is not implemented. For production use, retrain monthly.
- No transaction costs or slippage modelled.

---

## License

MIT License. Free to use, modify, and distribute with attribution.

---
