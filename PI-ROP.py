import os

import dotenv
import cv2
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf
from dotenv import load_dotenv
from tensorflow.keras.applications import ResNet50
# Import correto para a função de pré-processamento
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input

# Carrega as variáveis do arquivo .env para o ambiente
load_dotenv()


DATA_DIR = os.environ.get("DATA_DIR")
# IMAGES_DIR = os.path.join(DATA_DIR, os.environ.get("IMAGES_DIR"))

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

try:
    tf.TF_ENABLE_ONEDNN_OPTS = 0
    validation_split = 0.2
    seed = 1
    shuffle = True
    # Carregar dados de Treino
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=DATA_DIR,  # Usando  path fixo por enquanto
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=validation_split,
        seed=seed,
        label_mode='binary',
        subset="training",
        shuffle=shuffle

    )

    # Carregar dados de Validação
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=DATA_DIR,  # Usando  path fixo por enquanto
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary',
        validation_split=validation_split,
        seed=seed,
        subset="validation",
        shuffle=shuffle
    )

    rop_ausente = 0
    rop_presente = 1

    print("Classes encontradas:", train_dataset.class_names)

    # Otimizar a performance de carregamento
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomContrast(0.2),
        ],
        name="data_augmentation",
    )

    # 1. Carregar o Modelo Base (ResNet-50)
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3) #cores
    )

    # 2. Congelar o Modelo Base
    base_model.trainable = False

    # 3. Criar a nova "Cabeça" de Classificação
    # serve pra nao usar dados que o modelo já conhece

    inputs = tf.keras.Input(shape=(224, 224, 3), name="input_layer")

    x = data_augmentation(inputs)

    #  função de pre-processamento
    x = preprocess_input(x)

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(0.5)(x)  # Pode até aumentar para 0.7
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name="output_layer")(x)
    model = tf.keras.Model(inputs, outputs)
    model.summary()


    print("\nIniciando Fase 1: Treinando ...")

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

    # Salvar o modelo
    model.save("meu_modelo_rop_v1.h5")
    print("Modelo salvo em 'meu_modelo_rop_v1.h5'")

    # quanto mais perto o val_accuracy da accuracy melhor, e val_loss deve diminuir conforme as epocas

    test_loss, text_acc = model.evaluate(validation_dataset, verbose=2)
    print(test_loss, text_acc)
    model.summary()
except Exception as e:
    print("Ocorreu um erro:", e)