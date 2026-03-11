import cv2
from ultralytics import YOLO

# --- Configurações ---
MODELO_PATH = 'best3.pt'      # Caminho para o seu modelo treinado (.pt)
CAMERA_INDEX = 0             # 0 para webcam integrada, 1 ou 2 para USB
CONFIDENCE_THRESHOLD = 0.7    # Só mostra detecções com mais de 70% de certeza
# ---------------------

# 1. Carregar o modelo YOLOv8
try:
    print("Carregando o cérebro da IA...")
    model = YOLO(MODELO_PATH)
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    exit()

# 2. Iniciar a captura da webcam
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"Erro: Não foi possível abrir a câmera no índice {CAMERA_INDEX}.")
    exit()

print("Webcam aberta. Iniciando detecção em tempo real...")
print("Pressione [Q] para fechar a janela.")

while True:
    # 3. Ler um frame da câmera
    ret, frame = cap.read()
    if not ret:
        print("Erro: Não foi possível ler o frame.")
        break

    # 4. Fazer a detecção com o YOLOv8
    # Usamos o frame capturado e o threshold definido nas configurações
    results = model(frame, stream=True, conf=CONFIDENCE_THRESHOLD)

    # 5. Processar e Desenhar os Resultados
    # O YOLO já faz o trabalho pesado de identificar as classes
    for r in results:
        # O método .plot() desenha as caixas (bounding boxes) e nomes automaticamente
        annotated_frame = r.plot()

    # 6. Mostrar o frame na tela
    cv2.imshow("IA de Reconhecimento de Pecas - Offline", annotated_frame)

    # 7. Sair com a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Encerrando sistema...")
        break

# 8. Limpeza de recursos
cap.release()
cv2.destroyAllWindows()