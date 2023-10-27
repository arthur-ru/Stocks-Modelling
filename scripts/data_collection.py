import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

# URL de la page Wikipedia contenant la liste des symboles du S&P 500
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Envoyer une requête pour obtenir la page HTML
response = requests.get(url)

# Analyser la page HTML avec BeautifulSoup
soup = BeautifulSoup(response.text, "html.parser")

# Trouver le tableau contenant les symboles du S&P 500
table = soup.find("table", {"class": "wikitable"})

# Parcourir les lignes du tableau et extraire les symboles
sp500_symbols = []
for row in table.find_all("tr")[1:]:  # Commencez à partir de la deuxième ligne pour éviter l'en-tête
    columns = row.find_all("td")
    if len(columns) > 0:
        symbol = columns[0].text.strip()  # La première colonne contient les symboles
        sp500_symbols.append(symbol)

# Période de données historiques (depuis le 1er janvier 2020)
start_date = "2020-01-01"
end_date = "2023-01-01"

# Téléchargement des données
data = yf.download(sp500_symbols, start=start_date, end=end_date, group_by="ticker")

# Créez une liste pour stocker les symboles invalides
invalid_symbols = []

for symbol in sp500_symbols:
    try:
        # Essayez de télécharger les données pour chaque symbole
        data[symbol]
    except Exception as e:
        print(f"Erreur pour le symbole {symbol}: {e}")
        invalid_symbols.append(symbol)

# Sauvegarde des données dans un fichier CSV, index=False pour ne pas enregistrer l'index
data.to_csv("data/raw_data/data_stock.csv", index=False)

# Affichez la liste des symboles invalides
print("Symboles invalides :", invalid_symbols)