import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from modelos_avancados import PreditorPoseGRU, PreditorTransformer, PreditorTGNN

# Configurações
SEQ_LENGTH = 10
EPOCHS = 150
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Carregando poses.npy...")
try:
    poses = np.load("poses.npy")
except FileNotFoundError:
    print("ERRO: Rode o extrair_poses.py primeiro para gerar o poses.npy!")
    exit()

# Criar janelas deslizantes (X = sequencia passada, Y = próximo frame)
X, Y = [], []
for i in range(len(poses) - SEQ_LENGTH - 1):
    X.append(poses[i : i + SEQ_LENGTH])
    Y.append(poses[i + SEQ_LENGTH])

X = torch.tensor(np.array(X), dtype=torch.float32).to(device)
Y = torch.tensor(np.array(Y), dtype=torch.float32).to(device)

dataset = torch.utils.data.TensorDataset(X, Y)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Instanciar modelos
modelos = {
    "GRU": (PreditorPoseGRU().to(device), "modelo_gru.pth"),
    "Transformer": (PreditorTransformer().to(device), "modelo_transformer.pth"),
    "T-GNN": (PreditorTGNN().to(device), "modelo_tgnn.pth")
}

criterion = nn.MSELoss()

# Treinar cada modelo
for nome, (modelo, arquivo_saida) in modelos.items():
    print(f"\n--- Treinando {nome} ---")
    optimizer = optim.Adam(modelo.parameters(), lr=0.001)
    
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = modelo(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 30 == 0:
            print(f"Época {epoch+1}/{EPOCHS} | Loss: {epoch_loss/len(loader):.6f}")
            
    torch.save(modelo.state_dict(), arquivo_saida)
    print(f"[OK] {nome} salvo em {arquivo_saida}")