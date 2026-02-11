
---

# 🖐️ Real-Time Sign Language Detection System

A real-time sign language recognition system that detects and classifies hand gestures from live webcam video using **MediaPipe hand landmarks** and an **LSTM-based deep learning model**. The system supports temporal gesture modeling, confidence-based predictions, and real-time visual overlays.

---

## 📌 Project Overview

This project implements an end-to-end **computer vision + deep learning pipeline** for sign language recognition:

* **Hand detection & landmark extraction** using MediaPipe Hands
* **Automatic keypoint-based annotation** (21 hand landmarks per frame)
* **Temporal modeling** of gestures using 30-frame sequences
* **LSTM-based classification** for dynamic gesture recognition
* **Live webcam inference** with bounding boxes and confidence display

The system does **not rely on manual image annotations**; instead, it uses pretrained hand pose estimation to generate structured landmark inputs automatically.

---

## 🧠 Key Features

* Real-time gesture recognition from webcam feed
* Wrist-relative landmark normalization for translation invariance
* Sequence-based modeling (30 frames per gesture)
* Confidence thresholding to reduce false predictions
* Supports **both left and right hands** simultaneously
* Clean visual overlays with bounding boxes and labels

---

## 🛠️ Tech Stack

* **Language:** Python
* **Computer Vision:** OpenCV, MediaPipe
* **Deep Learning:** TensorFlow / Keras (LSTM)
* **Numerical Computing:** NumPy
* **Visualization:** OpenCV overlays

---

## 📂 Project Structure

```
Sign_Language_Detection_system/
│
├── data/                         # Collected gesture sequences (A–Z folders)
│   ├── A/
│   ├── B/
│   └── ...
│
├── landmarks_preprocessing.py    # Data loading, normalization, training
├── real_time_sign_detection.py   # Live webcam inference
├── best_model.h5                 # Best-performing trained model
├── sign_language_model.h5        # Final trained model
├── label_map.json                # Class label mapping
└── README.md
```

---

## 🔄 Pipeline Explanation

### 1️⃣ Data Collection

* Hand gestures are recorded using a webcam.
* MediaPipe extracts **21 3D landmarks per frame**.
* Each gesture is stored as a **30-frame sequence (30 × 63 features)**.

### 2️⃣ Preprocessing

* Landmarks are normalized relative to the wrist (landmark 0).
* This removes dependency on absolute hand position.
* Labels are encoded using one-hot encoding.

### 3️⃣ Model Training

* An **LSTM-based neural network** is trained to capture temporal motion patterns.
* Early stopping and model checkpointing are used to prevent overfitting.
* The best-performing model is saved as `best_model.h5`.

### 4️⃣ Real-Time Inference

* Webcam feed is processed frame-by-frame.
* Landmarks are buffered into rolling sequences.
* Predictions are displayed with bounding boxes and confidence scores.

---

## ▶️ How to Run

### 🔹 1. Install Dependencies

```bash
pip install opencv-python mediapipe tensorflow numpy
```

### 🔹 2. Train the Model

```bash
python landmarks_preprocessing.py
```

### 🔹 3. Run Real-Time Detection

```bash
python real_time_sign_detection.py
```

Press **`q`** to exit the webcam window.

---

## 📊 Model Details

* **Input Shape:** `(30, 63)`
* **Architecture:**

  * LSTM (64 units)
  * LSTM (128 units)
  * Dense + Softmax
* **Loss Function:** Categorical Cross-Entropy
* **Optimizer:** Adam

---

## 🚀 Results & Performance

* Stable real-time inference on standard laptop webcam
* Reduced misclassification using confidence thresholding
* Robust to moderate changes in hand position and orientation

*(Exact accuracy depends on dataset size and gesture similarity.)*

---

## 📌 Limitations

* Visually similar signs may require more training data
* Performance may drop in poor lighting conditions
* Currently supports single-hand gestures (extendable to words/sentences)

---

## 🔮 Future Improvements

* Add Transformer-based temporal modeling
* Integrate GPU acceleration (CUDA / TensorRT)
* Expand to word- and sentence-level sign recognition
* Include manually annotated datasets for comparison

---

## 👤 Author

**Rekibuddin Ansari**
GitHub: [Ansariricky](https://github.com/Ansariricky)

---

## ⭐ Acknowledgements

* MediaPipe for pretrained hand pose estimation
* TensorFlow/Keras for deep learning framework

---
