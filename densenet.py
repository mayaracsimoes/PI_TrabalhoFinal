import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
from dotenv import load_dotenv

load_dotenv()
DATA_DIR = os.environ.get("DATA_DIR")
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# NOME DIFERENTE
MODEL_NAME = "densenet121_rop.keras"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

try:
    print("\n--- INICIANDO EXPERIMENTO: DENSENET 121 ---")

    # 1. Dados
    train_ds = tf.keras.utils.image_dataset_from_directory(TRAIN_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                                                           label_mode='binary', shuffle=True)
    val_ds = tf.keras.utils.image_dataset_from_directory(VAL_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                                                         label_mode='binary', shuffle=False)
    test_ds = tf.keras.utils.image_dataset_from_directory(TEST_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                                                          label_mode='binary', shuffle=False)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 2. Modelo Base
    base_model = DenseNet121(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    # 3. Cabeça e Augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.3),
        tf.keras.layers.RandomZoom(0.3),
        tf.keras.layers.RandomContrast(0.25),
        tf.keras.layers.RandomBrightness(0.25),
    ])

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)

    # 4. Fase 1
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    print("Treinando Fase 1 (Congelado)...")
    history = model.fit(train_ds, epochs=12, validation_data=val_ds)

    # 5. Fase 2 (Fine Tuning)
    base_model.trainable = True
    fine_tune_at = 320  # Ajuste para DenseNet (Camada profunda)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    print("Treinando Fase 2 (Fine-Tuning)...")
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=[callback])

    model.save(MODEL_NAME)

    # 6. Avaliação e Matriz
    print("\n--- Gerando Matriz de Confusão (DenseNet121) ---")
    predictions = model.predict(test_ds)
    y_pred = (predictions > 0.5).astype(int).flatten()

    y_true = []
    for img, label in test_ds:
        y_true.extend(label.numpy())

    cm = confusion_matrix(y_true, y_pred)
    print(classification_report(y_true, y_pred, target_names=['Normal', 'ROP']))

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')  # Roxo para diferenciar
    plt.title(f'DenseNet121 (Acurácia: {np.mean(y_true == y_pred):.2%})')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    plt.show()

except Exception as e:
    print(e)