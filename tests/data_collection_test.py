import yfinance as yf
import pandas as pd
import concurrent.futures

# Lire les symboles du S&P 500 à partir de Wikipedia
sp500_symbols = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol'].tolist()

# Période de données historiques
start_date = "2020-01-01"
end_date = "2022-01-01"

# Fonction pour télécharger les données pour un symbole
def download_data(symbol):
    data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)
    if not data.empty:
        return symbol, data['Close']
    return None

# Utilisation de ProcessPoolExecutor pour paralléliser le téléchargement
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = executor.map(download_data, sp500_symbols)

# Filtrer les résultats pour exclure les symboles sans données
valid_data = {symbol: data for symbol, data in results if data is not None}

# Créer un DataFrame avec les données valides
valid_data_df = pd.DataFrame(valid_data)

# Sauvegarde des données dans un fichier CSV, index=False pour ne pas enregistrer l'index
valid_data_df.to_csv("data_stock_test.csv", index=False)
