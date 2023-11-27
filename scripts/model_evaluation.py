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

# After the training loop: evaluate the model on the test set
model.eval()        # Set the model to evaluation mode
total_loss = 0
total_samples = 0

with torch.no_grad():  # No need to track the gradients
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        # Resize labels for shape matching
        labels = labels.unsqueeze(1)

        # Compute the loss
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        total_samples += labels.size(0)

# Calculate the average loss
average_loss = total_loss / total_samples
print(f'Perte moyenne: {average_loss:.4f}')