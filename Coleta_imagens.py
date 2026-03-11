import cv2
import os
import time


# --- Configurações ---
CAMERA_INDEX = 0      # 0 é geralmente a webcam. Mude se for 1, 2, etc.
SAVE_PATH = "dataset_images" # Nome da pasta onde as imagens serão salvas
IMG_PREFIX = ""    # Prefixo do nome do arquivo (ex: "martelo", "alicate_bico")
# ---------------------
 
# 1. Criar a pasta de salvamento se ela não existir
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    print(f"Pasta '{SAVE_PATH}' criada.")
else:
    print(f"Salvando imagens na pasta '{SAVE_PATH}'.")

# 2. Iniciar a captura da webcam
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"Erro: Não foi possível abrir a câmera no índice {CAMERA_INDEX}.")
    exit()

print("\n--- Iniciando Coleta de Dados ---")
print("Pressione [ESPAÇO] para salvar uma imagem.")
print("Pressione [Q] para sair.")

img_counter = 0

# Tenta encontrar o último número de imagem salvo para não sobrescrever
# Isso é útil se você parar e reiniciar o script
files = os.listdir(SAVE_PATH)
if files:
    for f in files:
        if f.startswith(IMG_PREFIX) and f.endswith(".png"):
            try:
                num = int(f.replace(IMG_PREFIX, "").replace("_", "").replace(".png", ""))
                if num > img_counter:
                    img_counter = num
            except ValueError:
                continue
    if img_counter > 0:
        img_counter += 1 # Começa do próximo número
        print(f"Continuando a contagem de imagens a partir de {img_counter}.")


while True:
    # 3. Ler o frame da câmera
    ret, frame = cap.read()
    if not ret:
        print("Erro: Não foi possível ler o frame. Fim do stream?")
        break

    # 4. Mostrar o frame em uma janela
    # Coloca um texto na tela para sabermos o que fazer
    cv2.putText(frame, "Pressione [ESPACO] para salvar, [Q] para sair",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Coleta de Dados - Câmera ao Vivo", frame)

    # 5. Esperar por uma tecla
    key = cv2.waitKey(1) & 0xFF

    # 6. Lógica das teclas
    if key == ord('q'):
        print("Saindo...")
        break
    elif key == ord(' '): # Barra de espaço
        # Criar um nome de arquivo único
        img_name = f"{SAVE_PATH}/{IMG_PREFIX}_{img_counter}.png"

        #Salvar o frame atual
        cv2.imwrite(img_name, frame)
        print(f"IMAGEM SALVA: {img_name}")

        # Incrementar o contador
        img_counter += 1

# 7. Limpar tudo ao sair
print(f"Coleta finalizada. Total de {img_counter} imagens na pasta {SAVE_PATH}.")
cap.release()
cv2.destroyAllWindows()