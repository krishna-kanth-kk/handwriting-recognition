# 🖊️ Handwritten Digit Recognizer using CNN

A Convolutional Neural Network (CNN) trained on the famous **MNIST dataset** to recognize handwritten digits (0–9) with **~99% accuracy**.

---

## 📌 Project Overview

| Detail | Info |
|---|---|
| **Dataset** | MNIST (60,000 train / 10,000 test images) |
| **Model** | 2-Block CNN with BatchNorm + Dropout |
| **Accuracy** | ~99% on test set |
| **Framework** | TensorFlow / Keras |
| **Language** | Python 3.8+ |

---

## 🗂️ Project Structure

```
Project1_Handwritten_Digit_Recognizer/
├── train.py            # Train the CNN model
├── predict.py          # Load model & make predictions
├── requirements.txt    # Python dependencies
├── saved_model/        # Auto-created during training
│   └── best_model.keras
└── plots/              # Auto-created during training
    ├── training_history.png
    ├── sample_predictions.png
    └── random_predictions.png
```

---

## ⚙️ Setup & Installation

### 1. Clone / Download the project
```bash
git clone https://github.com/<your-username>/handwritten-digit-recognizer.git
cd handwritten-digit-recognizer
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1 — Train the Model
```bash
python train.py
```
- Downloads MNIST automatically
- Trains for up to 15 epochs (early stopping)
- Saves best model to `saved_model/best_model.keras`
- Generates training plots in `plots/`

### Step 2 — Make Predictions
```bash
# Predict on 10 random MNIST test images
python predict.py

# Predict on your own image (28x28 greyscale PNG)
python predict.py --image your_digit.png
```

---

## 🧠 Model Architecture

```
Input (28×28×1)
    ↓
Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout(0.25)
    ↓
Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout(0.25)
    ↓
Flatten → Dense(256) → BatchNorm → Dropout(0.5)
    ↓
Dense(10, softmax)  →  Output
```

---

## 📊 Results

| Metric | Value |
|---|---|
| Test Accuracy | ~99% |
| Test Loss | ~0.03 |

---

## 📚 References

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow Keras Docs](https://www.tensorflow.org/api_docs/python/tf/keras)
- Inspired by open-source CNN implementations on GitHub

---

## 🏷️ Tech Stack

`Python` · `TensorFlow` · `Keras` · `NumPy` · `Matplotlib`