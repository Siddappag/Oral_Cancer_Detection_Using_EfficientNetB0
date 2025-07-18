import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, recall_score
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight
from tensorflow.keras.applications.efficientnet import preprocess_input

# Paths
train_dir = r'C:\Users\shrey\OneDrive\Desktop\AI Chatbot\New folder\Oral_Cancer\Dataset\Train'
val_dir = r'C:\Users\shrey\OneDrive\Desktop\AI Chatbot\New folder\Oral_Cancer\Dataset\Validation'
test_dir = r'C:\Users\shrey\OneDrive\Desktop\AI Chatbot\New folder\Oral_Cancer\Dataset\Test'

# Image settings
image_size = (224, 224)  # EfficientNetB0 prefers 224x224
batch_size = 32

# Data generators
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, 
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input  # ← and here
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Compute class weights 
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# Build model: EfficientNetB0
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base


# After initial fit and before fine-tuning:
base_model.trainable = True
# Freeze only the first N layers (experiment N≈100)
for layer in base_model.layers[:100]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-8)

# Training
steps_per_epoch = max(1, len(train_generator))
validation_steps = max(1, len(val_generator))

history = model.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=50,
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weights
)

# Fine-tuning 
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

fine_tune_history = model.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=30,
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weights
)

# Save
model.save("oral_cancer_efficientnet_model.h5")

# Evaluate
test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
print(f"Test Accuracy: {test_acc:.4f}")

# Predictions
y_true = test_generator.classes
y_pred_prob = model.predict(test_generator)
y_pred = (y_pred_prob > 0.5).astype("int32")

def tta_predict(model, img, tta_steps=5):
    preds = []
    for _ in range(tta_steps):
        aug = train_datagen.random_transform(img)
        aug = np.expand_dims(aug, 0)
        preds.append(model.predict(aug)[0][0])
    return np.mean(preds)

# Run TTA predictions
tta_preds = []
for i in range(len(test_generator)):
    batch_x, _ = test_generator[i]
    for img in batch_x:
        tta_preds.append(tta_predict(model, img))

y_pred_prob = np.array(tta_preds)
y_pred_classes = (y_pred_prob > 0.5).astype("int32")

import tensorflow as tf

def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        mod_factor = tf.keras.backend.pow((1 - p_t), gamma)
        return tf.keras.backend.mean(alpha_factor * mod_factor * bce)
    return loss

model.compile(
    optimizer=Adam(1e-5),          # lower LR for fine-tuning
    loss=focal_loss(alpha=0.25, gamma=2.0),
    metrics=['accuracy']
)

print("\nClassification Report:\n", classification_report(y_true, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("AUC Score:", roc_auc_score(y_true, y_pred_prob))

best_thr, best_rec = 0.5, 0
for t in np.arange(0.2, 0.8, 0.05):
    preds = (y_pred_prob > t).astype(int)
    rec = recall_score(test_generator.classes, preds)
    if rec > best_rec:
        best_rec, best_thr = rec, t
print(f"\nBest recall {best_rec:.3f} at threshold {best_thr:.2f}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_pred_prob):.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
plt.figure(figsize=(6, 4))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()