import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Creating a GRU Network
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

# Path to the CSV file
file_path = './data/raw_data/data_stock.csv'

# Load data from the "data_stock.csv" CSV file
data = pd.read_csv(file_path)

# Select the 'AAPL' ticker (4th column) : 
# **TODO** : automatize this step for all tickers
current_stock_data = data.iloc[:252, 3]
test_stock_data = data.iloc[252:, 3]

# Convert to numpy array for easier processing
current_stock_data = current_stock_data.to_numpy()
test_stock_data = test_stock_data.to_numpy()

class StockDataset(Dataset):
    """ Personnalised dataset for stock prices """
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.window_size]
        label = self.data[idx + self.window_size]
        return torch.tensor(seq, dtype=torch.float).unsqueeze(-1), torch.tensor(label, dtype=torch.float)

# Define the window size for sequence creation
window_size = 60 # Example: use the last 60 days to predict the price of the next day

# Creation of the dataset and the dataloader
stock_dataset = StockDataset(current_stock_data, window_size)
stock_test_dataset = StockDataset(test_stock_data, window_size)
train_loader = DataLoader(stock_dataset, batch_size=32, shuffle=False)  # No need to shuffle for time series
test_loader = DataLoader(stock_test_dataset, batch_size=1, shuffle=False)

# Instanciation of the model
model = GRUNet(input_dim=1, hidden_dim=100, output_dim=1, num_layers=2)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Number of epochs
num_epochs = 93

# Training loop
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

# Save the model
torch.save(model.state_dict(), "models/model.pth")