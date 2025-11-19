from dotenv import load_dotenv
import os
import cv2

load_dotenv()

DATA_DIR = os.environ.get("DATA_DIR")

def clahe(image_path):
    img = cv2.imread(image_path, 1)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))

    img2 = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    cv2.imwrite(image_path, img2)

def pre_processar():
    arquivos = [f for f in os.listdir(DATA_DIR)
                if os.path.isfile(os.path.join(DATA_DIR, f))]
    if not arquivos:
        print("Nenhum arquivo encontrado no diretório.")
        return

    arquivos_processados = 0
    for arquivo in arquivos:
        arquivos_processados += 1
        clahe(os.path.join(DATA_DIR, arquivo))

    print(f"\nArquivos processados: {arquivos_processados}")

if __name__ == "__main__":
    pre_processar()