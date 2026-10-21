import torch
import torch.nn as nn
import math
import numpy as np

# A mesma escala usada na LSTM do seminário anterior
ESCALA_DELTA = 0.05

# ============================================================
# 1. GRU (Gated Recurrent Unit)
# ============================================================
class PreditorPoseGRU(nn.Module):
    def __init__(self, input_dim=34, hidden_dim=128, num_layers=2):
        super().__init__()
        # GRU é similar à LSTM, mas possui menos parâmetros e inferência mais rápida
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        ultimo = out[:, -1, :]
        # Prevê o deslocamento a partir do último frame conhecido
        return x[:, -1, :] + ESCALA_DELTA * self.fc(ultimo)

# ============================================================
# 2. TRANSFORMER TEMPORAL
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class PreditorTransformer(nn.Module):
    def __init__(self, input_dim=34, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.entrada_fc = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.saida_fc = nn.Linear(d_model, input_dim)

    def forward(self, x):
        # Mapeia dimensão 34 para d_model
        x_emb = self.entrada_fc(x)
        x_emb = self.pos_encoder(x_emb)
        out = self.transformer(x_emb)
        ultimo = out[:, -1, :]
        return x[:, -1, :] + ESCALA_DELTA * self.saida_fc(ultimo)

# ============================================================
# 3. T-GNN (Temporal Graph Neural Network)
# ============================================================
def criar_matriz_adjacencia():
    # 17 keypoints do YOLO
    adj = np.eye(17, dtype=np.float32)
    conexoes = [
        (0,1), (0,2), (1,3), (2,4), (5,7), (7,9), (6,8), (8,10),
        (5,6), (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)
    ]
    for a, b in conexoes:
        adj[a, b] = 1.0
        adj[b, a] = 1.0
    # Normalização básica de grafo
    soma_linhas = adj.sum(axis=1)
    adj_norm = adj / soma_linhas[:, np.newaxis]
    return torch.tensor(adj_norm)

class PreditorTGNN(nn.Module):
    def __init__(self, num_nodes=17, in_features=2, hidden_dim=64):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_features = in_features
        # Registra a matriz de adjacência como um buffer do PyTorch
        self.register_buffer('A', criar_matriz_adjacencia())
        
        # Convolução Espacial em Grafo (GCN)
        self.gcn_weight = nn.Linear(in_features, hidden_dim)
        
        # Convolução/Recorrência Temporal
        self.temporal = nn.GRU(num_nodes * hidden_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, num_nodes * in_features)

    def forward(self, x):
        batch, seq, _ = x.shape
        # Redimensiona para separar os nós (17) e suas coordenadas (2)
        x_graph = x.view(batch, seq, self.num_nodes, self.in_features)
        
        # Passo 1: Extração Espacial (GCN)
        # Multiplica a Matriz de Adjacência pelas características dos nós
        h_spatial = torch.einsum('vw, bswc -> bsvc', self.A, x_graph)
        h_spatial = self.gcn_weight(h_spatial)
        h_spatial = torch.relu(h_spatial)
        
        # Passo 2: Extração Temporal
        # Achata a dimensão espacial para passar na GRU
        h_temporal_in = h_spatial.view(batch, seq, -1)
        out, _ = self.temporal(h_temporal_in)
        ultimo = out[:, -1, :]
        
        return x[:, -1, :] + ESCALA_DELTA * self.fc(ultimo)