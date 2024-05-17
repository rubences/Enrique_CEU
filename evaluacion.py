import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.metrics import MeanIoU
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score, precision_score, recall_score, jaccard_score
import matplotlib.pyplot as plt

def load_test_data(image_dir, mask_dir, image_size=(256, 256)):
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

def evaluate_model(model, test_images, test_masks):
    test_dataset = create_dataset(test_images, test_masks, batch_size=1)
    results = model.evaluate(test_dataset)
    
    print(f'Loss: {results[0]}')
    print(f'Accuracy: {results[1]}')

    # Calcular predicciones
    y_pred = model.predict(test_dataset)
    y_pred_thresholded = y_pred.argmax(axis=-1).astype(np.uint8)
    y_true = test_masks.squeeze().astype(np.uint8)

    # Flatten masks for metric calculations
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred_thresholded.flatten()
    
    # Métricas
    precision = precision_score(y_true_flat, y_pred_flat, average='binary')
    recall = recall_score(y_true_flat, y_pred_flat, average='binary')
    f1 = f1_score(y_true_flat, y_pred_flat, average='binary')
    iou = jaccard_score(y_true_flat, y_pred_flat, average='binary')
    dice = 2 * (precision * recall) / (precision + recall)
    
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'IoU: {iou}')
    print(f'Dice Coefficient: {dice}')
    
    # Curva ROC y AUC
    fpr, tpr, _ = roc_curve(y_true_flat, y_pred_flat)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()
    
    # Curva Precision-Recall
    precision_vals, recall_vals, _ = precision_recall_curve(y_true_flat, y_pred_flat)
    plt.figure()
    plt.plot(recall_vals, precision_vals, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()
    
    # Matriz de Confusión
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Background', 'Lesion'], rotation=45)
    plt.yticks(tick_marks, ['Background', 'Lesion'])

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # Visualización de Segmentaciones
    for i in range(5):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(test_images[i].astype(np.uint8))

        plt.subplot(1, 3, 2)
        plt.title("True Mask")
        plt.imshow(y_true[i], cmap='gray')

        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(y_pred_thresholded[i], cmap='gray')
        plt.show()

# Ruta de las imágenes y máscaras de prueba
test_image_dir = '/workspaces/Enrique_CEU/test_imagenes_norm'
test_mask_dir = '/workspaces/Enrique_CEU/test_mascaras'

# Cargar las imágenes y máscaras de prueba
test_images, test_masks = load_test_data(test_image_dir, test_mask_dir)

# Cargar el mejor modelo guardado
best_model = tf.keras.models.load_model('model_best.keras', compile=False)
best_model.compile(optimizer=Adam(), 
                  loss=SparseCategoricalCrossentropy(),
                  metrics=[SparseCategoricalAccuracy()])

# Evaluar el modelo
evaluate_model(best_model, test_images, test_masks)
