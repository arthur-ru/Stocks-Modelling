import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

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
