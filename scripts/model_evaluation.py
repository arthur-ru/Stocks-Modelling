import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from model_training import data, model, StockDataset, window_size, device, criterion, training_row_index, index

test_stock_data = data.iloc[training_row_index:, index]
test_stock_data = test_stock_data.to_numpy()

stock_test_dataset = StockDataset(test_stock_data, window_size)
test_loader = DataLoader(stock_test_dataset, batch_size=1, shuffle=False)

# Après la boucle d'entraînement
model.eval()  # Passer en mode évaluation
total_loss = 0
total_samples = 0

with torch.no_grad():  # Pas de calcul de gradient
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        # Redimensionner les étiquettes pour la correspondance de forme
        labels = labels.unsqueeze(1)

        # Calculer la perte
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        total_samples += labels.size(0)

# Calculer la perte moyenne
average_loss = total_loss / total_samples
print(f'Perte moyenne: {average_loss:.4f}')