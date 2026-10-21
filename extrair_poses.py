import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIGURAÇÕES
# ============================================================

VIDEO_PATH = "video-ia.mp4"
MODEL_PATH = "yolo11n-pose.pt"
OUTPUT_PATH = "poses.npy"

NUM_LANDMARKS = 17
NUM_KEYPOINTS = NUM_LANDMARKS * 2


# ============================================================
# CARREGAR YOLO
# ============================================================

print("Carregando YOLO...")

model = YOLO(MODEL_PATH)

print("YOLO carregado!")


# ============================================================
# ABRIR VÍDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Não foi possível abrir o vídeo: {VIDEO_PATH}"
    )


# ============================================================
# EXTRAÇÃO
# ============================================================

poses = []

frame_count = 0

print("Extraindo poses...")


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    results = model(
        frame,
        verbose=False,
        conf=0.5
    )

    result = results[0]

    if result.keypoints is None:
        continue

    if len(result.keypoints) == 0:
        continue

    pontos = result.keypoints.xy.cpu().numpy()

    if len(pontos) == 0:
        continue

    # Primeira pessoa detectada
    pessoa = pontos[0]

    if pessoa.shape != (17, 2):
        continue

    # --------------------------------------------------------
    # Normalizar coordenadas
    # --------------------------------------------------------

    altura, largura = frame.shape[:2]

    pessoa = pessoa.astype(np.float32)

    pessoa[:, 0] /= largura
    pessoa[:, 1] /= altura

    # --------------------------------------------------------
    # Transformar (17, 2) em (34,)
    # --------------------------------------------------------

    vetor = pessoa.reshape(-1)

    poses.append(vetor)

    if frame_count % 50 == 0:
        print(
            f"Processados {frame_count} frames..."
        )


# ============================================================
# FINALIZAÇÃO
# ============================================================

cap.release()

poses = np.array(
    poses,
    dtype=np.float32
)

np.save(
    OUTPUT_PATH,
    poses
)


print()
print("=" * 60)
print("EXTRAÇÃO CONCLUÍDA")
print("=" * 60)

print(
    f"Frames analisados: {frame_count}"
)

print(
    f"Poses válidas: {len(poses)}"
)

print(
    f"Formato dos dados: {poses.shape}"
)

print(
    f"Arquivo salvo: {OUTPUT_PATH}"
)