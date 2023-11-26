import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU Layer
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Chemin du fichier CSV
file_path = '../data/raw_data/data_stock.csv'

# Charger les données
data = pd.read_csv(file_path)

# Sélectionner le ticker 'AAPL' (4e colonne)
aapl_data = data.iloc[:, 3]

# Convertir en numpy array pour faciliter le traitement
aapl_data = aapl_data.to_numpy()

class StockDataset(Dataset):
    """ Dataset personnalisé pour les données d'actions """
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.window_size]
        label = self.data[idx + self.window_size]
        return torch.tensor(seq, dtype=torch.float).unsqueeze(-1), torch.tensor(label, dtype=torch.float)

# Définir la taille de la fenêtre pour la création des séquences
window_size = 60  # Exemple : utiliser les prix des 60 derniers jours pour prédire le prix du lendemain

# Créer le dataset
stock_dataset = StockDataset(aapl_data, window_size)

# Créer le DataLoader
train_loader = DataLoader(stock_dataset, batch_size=32, shuffle=False)  # Pas de mélange pour les séries temporelles

for batch in train_loader:
    for data in batch:
        for data2 in data:
            print(data2)


# Instanciation du modèle
model = GRUNet(input_dim=1, hidden_dim=100, output_dim=1, num_layers=2)

# Fonction de perte et optimiseur
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

# Boucle d'entraînement (pseudo-code)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")