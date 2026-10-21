import cv2
import numpy as np
import torch
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Importando as novas arquiteturas
from modelos_avancados import PreditorPoseGRU, PreditorTransformer, PreditorTGNN

# ============================================================
# CONFIGURAÇÕES
# ============================================================
VIDEO_PATH = "video-ia.mp4" # Teste com o sintético e depois com o real
OUTPUT_PATH = "resultado_oclusao_avancada.mp4"
CSV_PATH = "resultados_comparativos.csv"
GRAPH_PATH = "grafico_erro_modelos.png"

YOLO_MODEL_PATH = "yolo11n-pose.pt"

# Pesos (você precisará rodar um script de treino para gerar estes arquivos)
GRU_MODEL_PATH = "modelo_gru.pth"
TRANSF_MODEL_PATH = "modelo_transformer.pth"
TGNN_MODEL_PATH = "modelo_tgnn.pth"

SEQ_LENGTH = 10
INPUT_DIM = 34
NUM_KEYPOINTS = 17

PILAR_X_MIN_NORM = 0.46
PILAR_X_MAX_NORM = 0.54

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("ESTIMAÇÃO DE POSE — GRU × TRANSFORMER × T-GNN")
print("=" * 60)

# ============================================================
# CONEXÕES DO ESQUELETO YOLO E FUNÇÕES AUXILIARES
# ============================================================
CONEXOES_POSE = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

def desenhar_esqueleto(frame, pose, width, height, cor, espessura=2, raio=4):
    pontos = []
    for i in range(NUM_KEYPOINTS):
        x = max(0, min(width - 1, int(pose[i * 2] * width)))
        y = max(0, min(height - 1, int(pose[i * 2 + 1] * height)))
        pontos.append((x, y))

    for a, b in CONEXOES_POSE:
        cv2.line(frame, pontos[a], pontos[b], cor, espessura, cv2.LINE_AA)

    for x, y in pontos:
        cv2.circle(frame, (x, y), raio, cor, -1, cv2.LINE_AA)

def extrair_pose(results, ultima_pose_px):
    r = results[0]
    if r.keypoints is None or len(r.keypoints) == 0: return None
    
    indice = int(r.boxes.conf.argmax().item()) if r.boxes is not None and len(r.boxes) > 0 else 0
    pontos = r.keypoints.xy[indice].cpu().numpy().astype(np.float64)
    if len(pontos) < NUM_KEYPOINTS: return None
    
    pontos = pontos[:NUM_KEYPOINTS].copy()
    faltando = np.all(pontos == 0.0, axis=1)
    
    if faltando.all(): return None
    if faltando.any():
        if ultima_pose_px is not None:
            pontos[faltando] = ultima_pose_px[faltando]
        else:
            pontos[faltando] = pontos[~faltando].mean(axis=0)
    return pontos

# ============================================================
# CARREGAR MODELOS
# ============================================================
print("\nCarregando Modelos...")
model_yolo = YOLO(YOLO_MODEL_PATH)

# Inicializando Modelos (Substitua pelos hyperparâmetros reais usados no seu treino)
modelo_gru = PreditorPoseGRU().to(device).eval()
modelo_transf = PreditorTransformer().to(device).eval()
modelo_tgnn = PreditorTGNN().to(device).eval()

# (Descomente após treinar os modelos e gerar os arquivos .pth)
# modelo_gru.load_state_dict(torch.load(GRU_MODEL_PATH, map_location=device))
# modelo_transf.load_state_dict(torch.load(TRANSF_MODEL_PATH, map_location=device))
# modelo_tgnn.load_state_dict(torch.load(TGNN_MODEL_PATH, map_location=device))

print("Todos os modelos carregados!")

# ============================================================
# PROCESSAMENTO DE VÍDEO
# ============================================================
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

x_min_pilar = int(PILAR_X_MIN_NORM * width)
x_max_pilar = int(PILAR_X_MAX_NORM * width)

historico_poses = []
ultima_pose_px = None
janela_gru, janela_transf, janela_tgnn = None, None, None

erros_gru, erros_transf, erros_tgnn = [], [], []
curvas = {}
dados_csv = []

frame_count = 0
em_oclusao = False
evento_id = 0
frames_no_evento = 0

print("Iniciando inferência frame a frame...")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1

    results = model_yolo(frame, verbose=False)
    
    cv2.rectangle(frame, (x_min_pilar, 0), (x_max_pilar, height), (30, 30, 30), -1)

    pontos = extrair_pose(results, ultima_pose_px)
    if pontos is None:
        out.write(frame)
        continue

    ultima_pose_px = pontos.copy()
    pose_real = np.zeros(INPUT_DIM, dtype=np.float32)
    pose_real[0::2] = pontos[:, 0] / width
    pose_real[1::2] = pontos[:, 1] / height

    centro_x = (pontos[11][0] + pontos[12][0]) / 2.0
    oclusao_ativa = (x_min_pilar <= centro_x <= x_max_pilar)

    # ========================================================
    # FORA DO PILAR (Rastreamento normal)
    # ========================================================
    if not oclusao_ativa:
        em_oclusao = False
        janela_gru = janela_transf = janela_tgnn = None

        historico_poses.append(pose_real.copy())
        if len(historico_poses) > SEQ_LENGTH:
            historico_poses.pop(0)

        desenhar_esqueleto(frame, pose_real, width, height, (0, 255, 0)) # VERDE = REAL

    # ========================================================
    # DENTRO DO PILAR (Predição Autorregressiva)
    # ========================================================
    else:
        if not em_oclusao:
            em_oclusao = True
            evento_id += 1
            frames_no_evento = 0
            if len(historico_poses) > 0:
                janela_base = list(historico_poses)
                while len(janela_base) < SEQ_LENGTH:
                    janela_base.insert(0, janela_base[0])
                
                # Inicializa janelas independentes para cada modelo
                base_array = np.array(janela_base[-SEQ_LENGTH:], dtype=np.float32)
                janela_gru = base_array.copy()
                janela_transf = base_array.copy()
                janela_tgnn = base_array.copy()
            else:
                janela_gru = janela_transf = janela_tgnn = None

        frames_no_evento += 1
        tempo_oclusao = frames_no_evento / float(fps)

        if janela_gru is None:
            out.write(frame)
            continue

        # 1. Inferência GRU
        in_gru = torch.tensor(janela_gru, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad(): pose_gru = modelo_gru(in_gru).cpu().numpy()[0]
        janela_gru = np.concatenate([janela_gru[1:], pose_gru[None, :]], axis=0)

        # 2. Inferência Transformer
        in_transf = torch.tensor(janela_transf, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad(): pose_transf = modelo_transf(in_transf).cpu().numpy()[0]
        janela_transf = np.concatenate([janela_transf[1:], pose_transf[None, :]], axis=0)

        # 3. Inferência T-GNN
        in_tgnn = torch.tensor(janela_tgnn, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad(): pose_tgnn = modelo_tgnn(in_tgnn).cpu().numpy()[0]
        janela_tgnn = np.concatenate([janela_tgnn[1:], pose_tgnn[None, :]], axis=0)

        # Avaliação de Erro
        erro_gru, erro_transf, erro_tgnn, pts = 0.0, 0.0, 0.0, 0
        for i in range(NUM_KEYPOINTS):
            real_x, real_y = pontos[i]
            if not (x_min_pilar <= real_x <= x_max_pilar): continue
            
            pts += 1
            gru_px = np.array([pose_gru[i*2] * width, pose_gru[i*2+1] * height])
            transf_px = np.array([pose_transf[i*2] * width, pose_transf[i*2+1] * height])
            tgnn_px = np.array([pose_tgnn[i*2] * width, pose_tgnn[i*2+1] * height])
            
            erro_gru += np.linalg.norm(pontos[i] - gru_px)
            erro_transf += np.linalg.norm(pontos[i] - transf_px)
            erro_tgnn += np.linalg.norm(pontos[i] - tgnn_px)

        if pts > 0:
            erro_gru /= pts
            erro_transf /= pts
            erro_tgnn /= pts

            erros_gru.append(erro_gru)
            erros_transf.append(erro_transf)
            erros_tgnn.append(erro_tgnn)

            if evento_id not in curvas: curvas[evento_id] = ([], [], [], [])
            curvas[evento_id][0].append(tempo_oclusao)
            curvas[evento_id][1].append(erro_gru)
            curvas[evento_id][2].append(erro_transf)
            curvas[evento_id][3].append(erro_tgnn)

            dados_csv.append([frame_count, evento_id, round(tempo_oclusao, 4), pts, erro_gru, erro_transf, erro_tgnn])

            # Desenho dos Esqueletos Inferidos
            desenhar_esqueleto(frame, pose_gru, width, height, (255, 0, 0), espessura=1)   # AZUL = GRU
            desenhar_esqueleto(frame, pose_transf, width, height, (128, 0, 128), espessura=1) # ROXO = Transformer
            desenhar_esqueleto(frame, pose_tgnn, width, height, (0, 165, 255), espessura=2) # LARANJA = T-GNN

            # HUD
            cv2.putText(frame, f"GRU: {erro_gru:.1f}px", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Transf: {erro_transf:.1f}px", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 128), 2)
            cv2.putText(frame, f"T-GNN: {erro_tgnn:.1f}px", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    out.write(frame)

cap.release()
out.release()

# Salvar CSV e Plotar Gráficos (Similar ao seu script original, mas com as 3 curvas)
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "evento_oclusao", "tempo_s", "pontos_ocultos", "erro_gru_px", "erro_transf_px", "erro_tgnn_px"])
    writer.writerows(dados_csv)

plt.figure(figsize=(12, 6))
for n, (evento, (t, e_gru, e_transf, e_tgnn)) in enumerate(curvas.items()):
    plt.plot(t, e_gru, color="blue", label="GRU" if n == 0 else None)
    plt.plot(t, e_transf, color="purple", label="Transformer" if n == 0 else None)
    plt.plot(t, e_tgnn, color="orange", linewidth=2, label="T-GNN" if n == 0 else None)

plt.xlabel("Tempo na oclusão (s)")
plt.ylabel("Erro médio (pixels)")
plt.title("Comparação de Estimação de Pose: GRU vs Transformer vs T-GNN")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(GRAPH_PATH, dpi=300)
plt.close()

print("\nPROCESSAMENTO CONCLUÍDO!")