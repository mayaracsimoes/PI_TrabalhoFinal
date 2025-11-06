import os
from dotenv import load_dotenv

import pandas as pd

# Carrega as variáveis do arquivo .env para o ambiente
load_dotenv()

# Paths
DATA_DIR = os.environ.get("DATA_DIR")
IMAGES_DIR = os.path.join(DATA_DIR, os.environ.get("IMAGES_DIR"))
try:
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Total de imagens no dataset: {len(image_files)}")

except Exception as e:
    print(" Dataset não importado", e)
    metadata_df = None
