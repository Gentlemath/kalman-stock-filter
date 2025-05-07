import yfinance as yf

def load_data(ticker = "AAPL", start = "2020-01-01", end= "2024-01-01"):
    df = yf.download(ticker, start = start, end = end)
    print(df.head())

    return df