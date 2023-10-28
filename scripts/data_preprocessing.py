import pandas as pd
import numpy as np
from scipy import stats

# Étape 1 : Charger les données depuis le fichier "data_stock.csv"
data = pd.read_csv("data/raw_data/data_stock.csv")

# Étape 2 : Extraire les valeurs de l'action "A" (deuxième colonne)
ticker_A_data = data.iloc[:, 1]

# Étape 3 : Calculer l'écart interquartile (IQR) des données de l'action "A"
Q1 = np.percentile(ticker_A_data, 25)
Q3 = np.percentile(ticker_A_data, 75)
IQR = Q3 - Q1

# Étape 4 : Calculer les limites supérieure et inférieure pour l'identification des outliers en utilisant la méthode IQR
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# Étape 7 : Identifier et afficher les outliers en fonction des deux méthodes
outliers_IQR = (ticker_A_data < lower_limit) | (ticker_A_data > upper_limit)

print("Outliers identifiés avec la méthode IQR :")
print(ticker_A_data[outliers_IQR])