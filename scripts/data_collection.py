import yfinance as yf
import pandas as pd
import numpy as np

# Lire les symboles du S&P 500 à partir de Wikipedia
sp500_symbols = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol'].tolist()

# Période de données historiques
start_date = "2020-01-01"
end_date = "2022-01-01"

# Liste pour stocker les symboles valides
valid_symbols = []

# Boucle pour filtrer les symboles valides
for symbol in sp500_symbols:
    data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)
    if not data.empty:
        valid_symbols.append(symbol)

# Obtenir les données pour les symboles valides
data = yf.download(valid_symbols, start=start_date, end=end_date, auto_adjust=True)['Close']

# Sauvegarde des données dans un fichier CSV, index=False pour ne pas enregistrer l'index
data.to_csv("data/raw_data/data_stock.csv", index=False)