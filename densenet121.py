import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Directorios de las imágenes normalizadas
melanoma_dir = '/Users/enriquegonzalezardura/Documents/DATASETS_copia_prueba/PH2Dataset/train/melanoma_A'
no_melanoma_dir = '/Users/enriquegonzalezardura/Documents/DATASETS_copia_prueba/PH2Dataset/train/no_melanoma_A'

# Verificación de archivos en los directorios
melanoma_files = [(os.path.join(melanoma_dir, f), 'melanoma') for f in os.listdir(melanoma_dir) if f.endswith('.png')]
no_melanoma_files = [(os.path.join(no_melanoma_dir, f), 'no_melanoma') for f in os.listdir(no_melanoma_dir) if f.endswith('.png')]

# Depuración: Imprimir rutas y verificación de accesibilidad
print(f'Melanoma Directory: {melanoma_dir}')
print(f'Non-Melanoma Directory: {no_melanoma_dir}')
print(f'Found {len(melanoma_files)} melanoma images')
print(f'Found {len(no_melanoma_files)} non-melanoma images')

# Comprobación si las rutas de los archivos existen
for filepath, label in melanoma_files + no_melanoma_files:
    if not os.path.exists(filepath):
        print(f'File not found: {filepath}')
    if not os.path.isfile(filepath):
        print(f'Not a file: {filepath}')

# Combinación de archivos y creación del DataFrame
all_files = melanoma_files + no_melanoma_files
df = pd.DataFrame(all_files, columns=['filename', 'label'])

# Depuración: Verificar que el DataFrame no esté vacío
print(f'Total images: {len(df)}')
print(df.head())

# Continuación con el entrenamiento si hay imágenes
if len(df) > 0:
    # Parámetros
    img_size = 224
    batch_size = 32
    epochs = 50

    # Generador de datos
    datagen = ImageDataGenerator()

    # Función para crear el modelo
    def create_model():
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = True

        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # K-fold cross validation
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_no = 1

    for train_index, val_index in kf.split(df['filename'], df['label']):
        train_data = df.iloc[train_index]
        val_data = df.iloc[val_index]

        train_generator = datagen.flow_from_dataframe(
            train_data,
            x_col='filename',
            y_col='label',
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True
        )

        val_generator = datagen.flow_from_dataframe(
            val_data,
            x_col='filename',
            y_col='label',
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )

        model = create_model()

        model_checkpoint = ModelCheckpoint(f'model_fold_{fold_no}.keras', save_best_only=True, monitor='val_loss')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=[model_checkpoint, early_stopping]
        )

        fold_no += 1
else:
    print("No images found. Please check the directories and image formats.")
