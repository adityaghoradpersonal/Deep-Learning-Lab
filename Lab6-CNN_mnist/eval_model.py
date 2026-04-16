import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

from sklearn.metrics import confusion_matrix, classification_report

import json
import pickle

# =========================
# STEP 1: LOAD DATA
# =========================
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize
X_test = X_test / 255.0

# Reshape for CNN
X_test = X_test.reshape(-1, 28, 28, 1)

# =========================
# STEP 2: LOAD SAVED MODEL
# =========================
model = load_model("mnist_cnn_model.h5")  
# or use: "mnist_cnn_model.keras"

print("Model loaded successfully!")

# =========================
# STEP 3: EVALUATE MODEL
# =========================
# One-hot encode test labels for evaluation
from tensorflow.keras.utils import to_categorical
y_test_cat = to_categorical(y_test, 10)

test_loss, test_acc = model.evaluate(X_test, y_test_cat)
print("Test Accuracy:", test_acc)

# =========================
# STEP 4: MAKE PREDICTIONS
# =========================
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# =========================
# STEP 5: CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_loaded.png")  # saved separately
plt.show()

# =========================
# STEP 6: CLASSIFICATION REPORT
# =========================
report = classification_report(y_test, y_pred)
print(report)

with open("classification_report_loaded.txt", "w") as f:
    f.write(report)

# =========================
# STEP 7: SAVE RESULTS AGAIN (OPTIONAL)
# =========================
results = {
    "test_accuracy": float(test_acc)
}

with open("results_loaded.json", "w") as f:
    json.dump(results, f)

with open("predictions_loaded.pkl", "wb") as f:
    pickle.dump(y_pred, f)

# =========================
# STEP 8: PREDICT SINGLE SAMPLE (DEMO)
# =========================
sample = X_test[0].reshape(1, 28, 28, 1)
prediction = model.predict(sample)
print("Sample Prediction:", np.argmax(prediction))
print("Actual Label:", y_test[0])