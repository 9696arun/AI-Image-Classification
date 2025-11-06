import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
test_dir = "final_dataset/test"
model_path = "model/best_model.h5"
results_dir = "results"

# Create folder to save results
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Load trained model
print("Loading trained model...")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.\n")

# Automatically detect input shape
input_shape = model.input_shape[1:3]
print(f"Detected model input size: {input_shape}\n")

# Prepare test data
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=input_shape,
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate model
print("Evaluating model on test data...\n")
loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}\n")

# Predictions
print("Generating predictions...")
y_pred = model.predict(test_generator, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Classification Report
print("\nClassification Report:\n")
report = classification_report(y_true, y_pred_classes, target_names=class_labels)
print(report)

# Confusion Matrix
print("Creating Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Save confusion matrix image before showing
save_path = os.path.join(results_dir, "confusion_matrix.png")
plt.tight_layout()
plt.savefig(save_path)
plt.show()

print(f"\nConfusion Matrix image saved at: {save_path}")
print("\n✅ Evaluation complete.")
