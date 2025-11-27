import os
import tensorflow as tf

import numpy as np
from dotenv import load_dotenv
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Carrega as variáveis do arquivo .env
load_dotenv()

# onde estão train, val, test
DATA_DIR = os.environ.get("DATA_DIR")

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

try:
    tf.TF_ENABLE_ONEDNN_OPTS = 0

    # 1. Dados de Treino
    print(f"Carregando treino de: {TRAIN_DIR}")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary',
        shuffle=True
    )

    # 2. Dados de Validação
    print(f"Carregando validação de: {VAL_DIR}")
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=VAL_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary',
        shuffle=False
    )

    # 3. Dados de Teste
    print(f"Carregando teste de: {TEST_DIR}")
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary',
        shuffle=False
    )

    print("Classes encontradas:", train_dataset.class_names)

    # Otimizar a performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    # Data Augmentation
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.3),
            tf.keras.layers.RandomZoom(0.3),
            tf.keras.layers.RandomContrast(0.25),
            tf.keras.layers.RandomBrightness(0.25),
        ],
        name="data_augmentation",
    )

    # --- MODELO ---
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3), name="input_layer")
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name="output_layer")(x)

    model = tf.keras.Model(inputs, outputs)
    model.summary()

    print("\nIniciando Treinamento...")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_dataset,
        epochs=15,
        validation_data=validation_dataset
    )

    print("Treinamento concluído!")

    model.save("meu_modelo_rop_v1.keras")
    print("Modelo salvo em 'meu_modelo_rop_v1.keras'")

    print("\n--- Iniciando Fase 2: Fine-Tuning ---")

    # 1. Descongelar o Modelo Base
    base_model.trainable = True

    # 2. Congelar as camadas iniciais (Fine-tune apenas no topo)
    # treinar apenas as últimas 35 (detectam texturas complexas/ROP)
    fine_tune_at = 140

    print(f"Número total de camadas no base_model: {len(base_model.layers)}")
    print(f"Congelando camadas até a {fine_tune_at}...")

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # 3. Recompilar com Learning Rate baixo
    # Usamos 1e-5 
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # 4. Callbacks para a Fase 2
    # Salvar o melhor modelo do Fine-Tuning separado
    checkpoint_ft = tf.keras.callbacks.ModelCheckpoint(
        filepath="meu_modelo_rop_finetuned.keras",
        save_best_only=True,
        monitor="val_loss",
        verbose=1
    )

    early_stop_ft = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # 5. Treinar a Fase 2
    # Total de épocas = Épocas da Fase 1 + Novas Épocas
    total_epochs = 15 + 20  # mais 20 épocas

    history_fine = model.fit(
        train_dataset,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1],  # Começa de onde a Fase 1 parou
        validation_data=validation_dataset,
        callbacks=[checkpoint_ft, early_stop_ft]
    )

    print("Fine-Tuning concluído!")

    print("\n--- Avaliação Final (Modelo Refinado) ---")
    # Recarregar o melhor modelo salvo pelo checkpoint
    best_model = tf.keras.models.load_model("meu_modelo_rop_finetuned.keras")

    loss, accuracy = best_model.evaluate(test_dataset)
    print(f"Loss Final: {loss:.4f}")
    print(f"Acurácia Final: {accuracy:.4f}")

    # --- AVALIAÇÃO FINAL (Usando a pasta 'test') ---
    print("\n--- Avaliação no conjunto de TESTE (dados nunca vistos) ---")
    test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
    print(f"Loss no Teste: {test_loss:.4f}")
    print(f"Acurácia no Teste: {test_acc:.4f}")

    # 1. Carregar o Modelo
    print(f"Carregando modelo: meu_modelo_rop_finetuned.keras")
    model = best_model

    # 2. Carregar dados de Teste
    print("Carregando imagens de teste...")
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary',
        shuffle=False
    )

    class_names = test_dataset.class_names
    print(f"Classes: {class_names}")

    # 3. Fazer Predições
    print("Gerando predições...")
    # Pega as predições (probabilidades entre 0 e 1)
    predictions = model.predict(test_dataset)

    # Converte para 0 ou 1 (se > 0.5 é classe 1, senão classe 0)
    predicted_labels = (predictions > 0.5).astype(int).flatten()

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
    print("Ocorreu um erro:", e)
