import yfinance as yf
import pandas as pd
import numpy as np

# Lire les symboles du S&P 500 à partir de Wikipedia
sp500_symbols = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol'].tolist()

# Période de données historiques
start_date = "2018-01-01"
end_date = "2020-01-01"

# Liste pour stocker les symboles valides et vides
valid_symbols = []
invalid_symbols = []

# Boucle pour filtrer les symboles valides et vides
for symbol in sp500_symbols:
    try:
        data = yf.download(symbol, period="1mo",start=start_date, end=end_date, auto_adjust=True)
        if not data.empty:
            valid_symbols.append(symbol)
        else:
            invalid_symbols.append(symbol)
    except Exception as e:
        print(f"Failed download for {symbol}: {e}")
        invalid_symbols.append(symbol)

# Obtenir les données pour les symboles valides
data = yf.download(valid_symbols, period="1d", start=start_date, end=end_date, auto_adjust=True)['Close']
#Ajouter la colonne des jours
data = data.reset_index()

# Sauvegarde des données dans un fichier CSV, index=False pour ne pas enregistrer l'index
data.to_csv("data/raw_data/data_stock.csv", index=False)

# Enregistrez les symboles pour lesquels aucune donnée n'était disponible dans un fichier ou imprimez-les
with open("data/raw_data/invalid_symbols.txt", "w") as f:
    f.write("\n".join(invalid_symbols))