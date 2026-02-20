import numpy as np
import matplotlib.pyplot as plt
import os

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ─── 1. LOAD & PREPROCESS DATA ────────────────────────────────────────────────

print("Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape to (samples, height, width, channels) and normalise to [0, 1]
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test  = X_test.reshape(-1,  28, 28, 1).astype("float32") / 255.0

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat  = to_categorical(y_test,  10)

print(f"Training samples : {X_train.shape[0]}")
print(f"Test samples     : {X_test.shape[0]}")

# ─── 2. BUILD CNN MODEL ───────────────────────────────────────────────────────

def build_model():
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Classifier head
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax'),
    ])
    return model

model = build_model()
model.summary()

# ─── 3. COMPILE ───────────────────────────────────────────────────────────────

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ─── 4. CALLBACKS ─────────────────────────────────────────────────────────────

os.makedirs("saved_model", exist_ok=True)

callbacks = [
    ModelCheckpoint(
        filepath="saved_model/best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]

# ─── 5. TRAIN ─────────────────────────────────────────────────────────────────

print("\nTraining model...")
history = model.fit(
    X_train, y_train_cat,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# ─── 6. EVALUATE ──────────────────────────────────────────────────────────────

test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n✅ Test Accuracy : {test_acc * 100:.2f}%")
print(f"   Test Loss     : {test_loss:.4f}")

# ─── 7. SAVE TRAINING PLOTS ───────────────────────────────────────────────────

os.makedirs("plots", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['accuracy'],     label='Train Acc')
axes[0].plot(history.history['val_accuracy'], label='Val Acc')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['loss'],     label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Val Loss')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("plots/training_history.png", dpi=150)
print("📊 Training plot saved → plots/training_history.png")

# ─── 8. SAMPLE PREDICTIONS ────────────────────────────────────────────────────

predictions = np.argmax(model.predict(X_test[:25], verbose=0), axis=1)

fig, axes = plt.subplots(5, 5, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
    color = 'green' if predictions[i] == y_test[i] else 'red'
    ax.set_title(f"Pred:{predictions[i]}  True:{y_test[i]}", fontsize=7, color=color)
    ax.axis('off')

plt.suptitle("Sample Predictions (Green=Correct, Red=Wrong)", fontsize=10)
plt.tight_layout()
plt.savefig("plots/sample_predictions.png", dpi=150)
print("🖼️  Sample predictions saved → plots/sample_predictions.png")
plt.show()