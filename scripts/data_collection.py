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

# Afficher les symboles
for symbol in sp500_symbols:
    print(symbol)
