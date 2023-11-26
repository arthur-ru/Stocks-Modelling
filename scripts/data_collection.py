import yfinance as yf
import pandas as pd
import numpy as np

# Read S&P 500 symbols from Wikipedia
sp500_symbols = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol'].tolist()

# Historical data period
start_date = "2018-01-01"
end_date = "2020-01-01"

# Lists to store valid and empty symbols (meaning no data was available)
valid_symbols = []
invalid_symbols = []

# Loop to filter valid and empty symbols
for symbol in sp500_symbols:
    try:
        # Period is set to 1mo to avoid overloading the Yahoo Finance API
        data = yf.download(symbol, period="1mo",start=start_date, end=end_date, auto_adjust=True)
        if not data.empty:
            valid_symbols.append(symbol)
        else:
            invalid_symbols.append(symbol)
    except Exception as e:
        print(f"Failed download for {symbol}: {e}")
        invalid_symbols.append(symbol)

# Get data for valid symbols (period is set to 1d to get daily data)
data = yf.download(valid_symbols, period="1d", start=start_date, end=end_date, auto_adjust=True)['Close']
# Add the date column
data = data.reset_index()

# Save data to a CSV file, index=False to not save the index (the dates)
data.to_csv("data/raw_data/data_stock.csv", index=False)

# Save symbols for which no data was available to a file called invalid_symbols.txt
with open("data/raw_data/invalid_symbols.txt", "w") as f:
    f.write("\n".join(invalid_symbols))