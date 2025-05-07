# kalman-stock-filter
A lightweight Python framework for testing various filters

# Kalman Stock Filter

A lightweight Python framework for testing various filters—Kalman Filter, Extended Kalman Filter (EKF), Bayesian Filter, and Particle Filter—on stock price prediction tasks.

## 📦 Features

- Modular structure for easy experimentation
- Built-in implementations of:
  - Kalman Filter (via `pykalman`)
  - Bayesian Filter (window-based)
  - Extended Kalman Filter (EKF-ready)
- Designed for univariate stock price time series
- Compatible with real-world data (e.g., via `yfinance`)

## 🚀 Getting Started

### Install Dependencies

Create a virtual environment and install required packages:

```bash
pip install -r requirements.txt
```

### Run Example

```bash
python main.py
```

## 📁 Project Structure

```
kalman-stock-filter/
│
├── main.py                # Entry point
├── filters.py             # Filter implementations
├── tester.py              # Evaluation class
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── ...
```

## 📊 Dependencies

- numpy
- pandas
- matplotlib
- yfinance
- pykalman
- seaborn

## 📜 License

MIT License

---

🔍 For academic use or extensions (e.g., multivariate filtering, portfolio signals), contributions are welcome!
