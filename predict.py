import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# ─── ARGUMENT PARSER ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Handwritten Digit Predictor")
parser.add_argument("--image", type=str, default=None,
                    help="Path to a custom 28×28 greyscale PNG/JPG image")
args = parser.parse_args()

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────

MODEL_PATH = "saved_model/best_model.keras"
print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!\n")

# ─── PREDICT ──────────────────────────────────────────────────────────────────

if args.image:
    # Custom image provided by user
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    img = load_img(args.image, color_mode="grayscale", target_size=(28, 28))
    img_array = img_to_array(img).reshape(1, 28, 28, 1).astype("float32") / 255.0

    pred_probs = model.predict(img_array, verbose=0)[0]
    predicted  = np.argmax(pred_probs)
    confidence = pred_probs[predicted] * 100

    print(f"Predicted Digit : {predicted}")
    print(f"Confidence      : {confidence:.2f}%")

    plt.figure(figsize=(5, 5))
    plt.imshow(img_array.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted}  ({confidence:.1f}%)", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("plots/custom_prediction.png", dpi=150)
    plt.show()

else:
    # Random MNIST test samples
    (_, _), (X_test, y_test) = mnist.load_data()
    X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    indices = np.random.choice(len(X_test), 10, replace=False)
    samples = X_test[indices]
    labels  = y_test[indices]

    pred_probs   = model.predict(samples, verbose=0)
    predictions  = np.argmax(pred_probs, axis=1)
    confidences  = np.max(pred_probs, axis=1) * 100

    print(f"{'Index':<6} {'True':>4} {'Pred':>4} {'Confidence':>10} {'Correct':>8}")
    print("-" * 40)
    for i, idx in enumerate(indices):
        correct = "✅" if predictions[i] == labels[i] else "❌"
        print(f"{idx:<6} {labels[i]:>4} {predictions[i]:>4} {confidences[i]:>9.1f}%  {correct}")

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].reshape(28, 28), cmap='gray')
        color = 'green' if predictions[i] == labels[i] else 'red'
        ax.set_title(f"P:{predictions[i]} T:{labels[i]}\n{confidences[i]:.0f}%",
                     fontsize=9, color=color)
        ax.axis('off')
    plt.suptitle("Handwritten Digit Predictions", fontsize=12)
    plt.tight_layout()
    plt.savefig("plots/random_predictions.png", dpi=150)
    plt.show()
    print("\n📊 Plot saved → plots/random_predictions.png")