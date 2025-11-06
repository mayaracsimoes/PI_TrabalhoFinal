import os
import pandas as pd
from glob import glob

# Paths
DATA_DIR = 'data/archive/'
IMAGES_DIR = os.path.join(DATA_DIR, 'images_stack_without_captions/images_stack_without_captions')
try:
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Total de imagens no dataset: {len(image_files)}")

except Exception as e:
    print(" Dataset não importado", e)
    metadata_df = None
