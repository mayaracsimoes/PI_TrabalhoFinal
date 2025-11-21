import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from dotenv import load_dotenv

load_dotenv()

# Configurações
DATA_DIR = os.environ.get("DATA_DIR")
TEST_DIR = os.path.join(DATA_DIR, 'test')  # Garanta que esse caminho existe
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_PATH = "meu_modelo_rop_finetuned.keras"  # O seu melhor modelo

try:
    # 1. Carregar o Modelo
    print(f"Carregando modelo: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 2. Carregar dados de Teste (IMPORTANTE: shuffle=False para manter a ordem)
    print("Carregando imagens de teste...")
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary',
        shuffle=False  # CRUCIAL: Não embaralhar para comparar com os labels verdadeiros
    )

    class_names = test_dataset.class_names
    print(f"Classes: {class_names}")

    # 3. Fazer Predições
    print("Gerando predições...")
    # Pega as predições (probabilidades entre 0 e 1)
    predictions = model.predict(test_dataset)

    # Converte para 0 ou 1 (se > 0.5 é classe 1, senão classe 0)
    predicted_labels = (predictions > 0.3).astype(int).flatten()

    # 4. Pegar os Labels Verdadeiros
    true_labels = []
    for images, labels in test_dataset:
        true_labels.extend(labels.numpy())
    true_labels = np.array(true_labels).flatten().astype(int)

    # 5. Gerar Matriz de Confusão
    cm = confusion_matrix(true_labels, predicted_labels)

    # 6. Plotar a Matriz
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predito pelo Modelo')
    plt.ylabel('Real (Verdadeiro)')
    plt.title('Matriz de Confusão - Detecção de ROP')
    plt.show()

    # 7. Imprimir Relatório Completo (Precisão, Recall, F1-Score)
    print("\n--- Relatório de Classificação ---")
    print(classification_report(true_labels, predicted_labels, target_names=class_names))

except Exception as e:
    print(f"Erro: {e}")