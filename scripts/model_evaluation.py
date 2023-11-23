import torch
import torch.nn as nn

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

# Instanciation du modèle
model = GRUNet(input_dim=1, hidden_dim=100, output_dim=1, num_layers=2)

# Fonction de perte et optimiseur
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

# Boucle d'entraînement (pseudo-code)
for epoch in range(num_epochs):
    for batch in train_loader:
        # Obtention des données et des étiquettes
        # Propagation avant
        # Calcul de la perte
        # Propagation arrière et optimisation
