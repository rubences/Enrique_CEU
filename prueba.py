import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

def load_images_and_masks(image_dir, mask_dir, image_size=(256, 256)):
    images = []
    masks = []
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    for img_file in image_files:
        mask_file = img_file.replace("aug", "lesion_aug")
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        if mask_file in mask_files and os.path.exists(img_path) and os.path.exists(mask_path):
            image = img_to_array(load_img(img_path, target_size=image_size))
            mask = img_to_array(load_img(mask_path, color_mode='grayscale', target_size=image_size))
            mask = mask / 255.0  # Normalize masks to have values between 0 and 1
            masks.append(np.round(mask))
            images.append(image)

    if not images or not masks:
        raise ValueError("No matching image and mask files found.")

    return np.array(images), np.array(masks)

# Load the images and masks
image_dir = './train_imagenes_norm_A'
mask_dir = './train_mascaras_A'
images, masks = load_images_and_masks(image_dir, mask_dir)

def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    # Encoder
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(2, (1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer=Adam(), 
                  loss=SparseCategoricalCrossentropy(),
                  metrics=[SparseCategoricalAccuracy()])

    return model

# Preparando la validación cruzada
num_folds = 10
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Variables para almacenar los resultados
fold_no = 1
loss_per_fold = []
acc_per_fold = []

# Validación cruzada de K-Fold
for train_index, test_index in kf.split(images):
    # Dividir los datos en entrenamiento y validación
    train_images, val_images = images[train_index], images[test_index]
    train_masks, val_masks = masks[train_index], masks[test_index]

    # Crear un nuevo modelo para evitar la contaminación entre pliegues
    model = unet_model(input_size=(256, 256, 3))

    # Entrenamiento del modelo
    print(f'Training for fold {fold_no}')
    model.fit(train_images, train_masks, batch_size=2, epochs=20, verbose=1)

    # Evaluación del modelo en el conjunto de validación
    scores = model.evaluate(val_images, val_masks, verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Liberar memoria explícitamente
    del model
    tf.keras.backend.clear_session()

    # Incrementar el número de pliegue
    fold_no += 1

# Promedio y desviación estándar de la pérdida y precisión
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')