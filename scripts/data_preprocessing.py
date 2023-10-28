import pandas as pd
import numpy as np

# Étape 1 : Charger les données depuis le fichier "data_stock.csv"
data = pd.read_csv("data/raw_data/data_stock.csv")

# Récupérer les noms des colonnes (à l'exception de la colonne des dates)
column_names = data.columns[1:]  # Commencer depuis la deuxième colonne

# Créer un dictionnaire pour stocker les outliers
outliers_data = {}

# Étape 2 : Itérer sur chaque colonne et identifier les outliers en utilisant la méthode IQR
for column_name in column_names:
    ticker_data = data[column_name]
    Q1 = np.percentile(ticker_data, 25)
    Q3 = np.percentile(ticker_data, 75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    outliers_IQR = (ticker_data < lower_limit) | (ticker_data > upper_limit)

    # Ajouter les outliers de cette colonne au dictionnaire
    outliers_data[column_name] = ticker_data[outliers_IQR]

    # Afficher les outliers pour cette colonne
    print(f"Outliers identifiés avec la méthode IQR pour {column_name}:")
    print(ticker_data[outliers_IQR])

# Créer un DataFrame à partir du dictionnaire des outliers
outliers_df = pd.DataFrame(outliers_data)

# Ajouter les autres colonnes du DataFrame original (y compris la colonne des dates)
outliers_df = pd.concat([data.iloc[:, 0], outliers_df], axis=1)

#Enregistrer les données dans un fichier CSV
outliers_df.to_csv("data/raw_data/outliers.csv", index=False)