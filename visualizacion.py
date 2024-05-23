from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
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
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import mixed_precision

# Establecer la política de precisión mixta
mixed_precision.set_global_policy('mixed_float16')

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
            mask = mask / 255.0  # Normalizar máscaras para tener valores entre 0 y 1
            masks.append(np.round(mask))
            images.append(image)

    if not images or not masks:
        raise ValueError("No matching image and mask files found.")

    return np.array(images), np.array(masks)

def create_dataset(images, masks, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.map(lambda img, mask: (tf.image.per_image_standardization(img), mask), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    # Encoder
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(32, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(16, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(16, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(2, (1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer=Adam(), 
                  loss=SparseCategoricalCrossentropy(),
                  metrics=[SparseCategoricalAccuracy()])

    return model

# Ruta de las imágenes y máscaras
image_dir = 'train_imagenes_norm_A'
mask_dir = 'train_mascaras_A'

# Cargar las imágenes y máscaras
images, masks = load_images_and_masks(image_dir, mask_dir)

# Preparando la validación cruzada
num_folds = 10
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Variables para almacenar los resultados
fold_no = 1
loss_per_fold = []
acc_per_fold = []

# Estrategia de distribución
strategy = tf.distribute.MirroredStrategy()

# Callbacks
callbacks = [
    ModelCheckpoint('model_best.keras', save_best_only=True, monitor='val_loss', mode='min'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
]

# Validación cruzada de K-Fold
for train_index, test_index in kf.split(images):
    # Dividir los datos en entrenamiento y validación
    train_images, val_images = images[train_index], images[test_index]
    train_masks, val_masks = masks[train_index], masks[test_index]

    train_dataset = create_dataset(train_images, train_masks, batch_size=1)
    val_dataset = create_dataset(val_images, val_masks, batch_size=1)

    with strategy.scope():
        # Crear un nuevo modelo para evitar la contaminación entre pliegues
        model = unet_model(input_size=(256, 256, 3))

        # Entrenamiento del modelo
        print(f'Training for fold {fold_no}')
        model.fit(train_dataset, validation_data=val_dataset, epochs=20, callbacks=callbacks, verbose=1)

    # Evaluación del modelo en el conjunto de validación
    scores = model.evaluate(val_dataset, verbose=0)
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

# Función para mostrar imágenes con sus máscaras
def display_images_with_masks(images, masks, predictions=None, n=5):
    plt.figure(figsize=(20, 10))
    for i in range(n):
        plt.subplot(3, n, i + 1)
        plt.imshow(images[i].astype('uint8'))
        plt.title('Image')
        plt.axis('off')

        plt.subplot(3, n, i + 1 + n)
        plt.imshow(masks[i].squeeze(), cmap='gray')
        plt.title('Mask')
        plt.axis('off')

        if predictions is not None:
            plt.subplot(3, n, i + 1 + 2 * n)
            plt.imshow(predictions[i].squeeze(), cmap='gray')
            plt.title('Prediction')
            plt.axis('off')
    plt.show()

# Mostrar algunas imágenes con sus máscaras reales
display_images_with_masks(images[:5], masks[:5])

# Generar algunas predicciones con el modelo entrenado
model = unet_model(input_size=(256, 256, 3))
model.fit(images, masks, batch_size=2, epochs=20, verbose=1)
predictions = model.predict(images[:5])

# Mostrar las imágenes, máscaras reales y predicciones
display_images_with_masks(images[:5], masks[:5], predictions[:5])

# Graficar la precisión y la pérdida por pliegue
def plot_kfold_metrics(acc_per_fold, loss_per_fold):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(acc_per_fold) + 1), acc_per_fold, 'o-')
    plt.title('Accuracy per fold')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy (%)')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(loss_per_fold) + 1), loss_per_fold, 'o-')
    plt.title('Loss per fold')
    plt.xlabel('Fold')
    plt.ylabel('Loss')
    plt.grid()

    plt.show()

# Graficar las métricas de evaluación
plot_kfold_metrics(acc_per_fold, loss_per_fold)

# Función para mostrar las curvas de entrenamiento y validación
def plot_training_history(history, fold_no):
    plt.figure(figsize=(12, 4))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.title(f'Model accuracy for fold {fold_no}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.title(f'Model loss for fold {fold_no}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')

    plt.show()

# Función para mostrar la matriz de confusión
def plot_confusion_matrix(y_true, y_pred, fold_no):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix for fold {fold_no}')
    plt.show()

# Función para mostrar histogramas de las métricas
def plot_metrics_histograms(acc_per_fold, loss_per_fold):
    plt.figure(figsize=(12, 6))

    # Histogram of accuracy
    plt.subplot(1, 2, 1)
    sns.histplot(acc_per_fold, bins=10, kde=True)
    plt.title('Accuracy Distribution')
    plt.xlabel('Accuracy (%)')

    # Histogram of loss
    plt.subplot(1, 2, 2)
    sns.histplot(loss_per_fold, bins=10, kde=True)
    plt.title('Loss Distribution')
    plt.xlabel('Loss')

    plt.show()

# Variables para almacenar las predicciones y las métricas de cada pliegue
histories = []
all_y_true = []
all_y_pred = []

# Validación cruzada de K-Fold con visualizaciones adicionales
for train_index, test_index in kf.split(images):
    # Dividir los datos en entrenamiento y validación
    train_images, val_images = images[train_index], images[test_index]
    train_masks, val_masks = masks[train_index], masks[test_index]

    # Crear un nuevo modelo para evitar la contaminación entre pliegues
    model = unet_model(input_size=(256, 256, 3))

    # Entrenamiento del modelo
    print(f'Training for fold {fold_no}')
    history = model.fit(train_images, train_masks, batch_size=2, epochs=20, verbose=1)
    histories.append(history)

    # Evaluación del modelo en el conjunto de validación
    scores = model.evaluate(val_images, val_masks, verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Predicciones en el conjunto de validación
    val_predictions = model.predict(val_images)
    val_predictions = np.argmax(val_predictions, axis=-1)

    # Almacenar las predicciones y las verdaderas etiquetas
    all_y_true.append(val_masks)
    all_y_pred.append(val_predictions)

    # Visualizar las curvas de entrenamiento y validación
    plot_training_history(history, fold_no)

    # Visualizar la matriz de confusión
    plot_confusion_matrix(val_masks, val_predictions, fold_no)

    # Liberar memoria explícitamente
    del model
    tf.keras.backend.clear_session()

    # Incrementar el número de pliegue
    fold_no += 1

# Graficar los histogramas de las métricas
plot_metrics_histograms(acc_per_fold, loss_per_fold)

# Promedio y desviación estándar de la pérdida y precisión
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
