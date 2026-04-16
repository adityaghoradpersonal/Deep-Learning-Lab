import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix, classification_report

import json

# Set your dataset path
data_dir = "D:/1 Projects/Computer Science/Degree/Degree 6th Semester/Deep Learning/Deep-Learning-Lab/Lab7-CNN_PlantDisease/PlantVillage"   # folder containing subfolders of classes

# Kaggle dataset path (auto-mounted)
# data_dir = "/kaggle/input/plantvillage-dataset/color"

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
#   For Kaggle
#   epochs=10     
)

val_loss, val_acc = model.evaluate(val_generator)
print("Validation Accuracy:", val_acc)

y_pred_probs = model.predict(val_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

y_true = val_generator.classes

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("plant_confusion_matrix.png")
plt.show()

report = classification_report(y_true, y_pred)
print(report)

with open("plant_classification_report.txt", "w") as f:
    f.write(report)

model.save("plant_disease_model.keras")

results = {
    "validation_accuracy": float(val_acc)
}

with open("plant_results.json", "w") as f:
    json.dump(results, f)