import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import json

# === Path & Config ===
DATA_PATH = "E:\\projects\\Sign_Language_Detection_system\\data"
SEQ_LENGTH = 30
LABELS = sorted(os.listdir(DATA_PATH))  # auto detect labels

# === Normalize Each Sequence ===
def normalize_sequence(seq):
    """
    Normalize each frame with respect to wrist (landmark 0).
    Input shape: (30, 63) → Output shape: (30, 63)
    """
    seq = seq.reshape(SEQ_LENGTH, 21, 3)
    wrist = seq[:, 0, :]  # wrist landmark per frame
    norm_seq = seq - wrist[:, np.newaxis, :]  # normalize each frame
    return norm_seq.reshape(SEQ_LENGTH, 63)

# === Load Data ===
def load_data():
    sequences, labels = [], []
    for label in LABELS:
        label_path = os.path.join(DATA_PATH, label)
        for seq_file in os.listdir(label_path):
            seq_path = os.path.join(label_path, seq_file)
            seq = np.load(seq_path)
            seq = normalize_sequence(seq)
            sequences.append(seq)
            labels.append(label)
    return np.array(sequences), np.array(labels)

# === Encode Labels ===
def encode_labels(labels):
    label_to_index = {label: idx for idx, label in enumerate(LABELS)}
    y = np.array([label_to_index[label] for label in labels])
    return to_categorical(y, num_classes=len(LABELS))

# === Main ===
if __name__ == "__main__":
    print("[INFO] Loading and preprocessing data...")
    X, y_raw = load_data()
    y = encode_labels(y_raw)

    print(f"[INFO] Data shape: {X.shape}")   # (num_samples, 30, 63)
    print(f"[INFO] Labels shape: {y.shape}") # (num_samples, num_classes)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y_raw
    )

    # Save preprocessed data
    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)

    # Save label mapping
    with open("label_map.json", "w") as f:
        json.dump(LABELS, f)

    print("[INFO] Training model...")
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, 63)))
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', save_best_only=True)
    ]

    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=50,
              batch_size=16,
              callbacks=callbacks)

    model.save('sign_language_model.h5')
    print("[INFO] Training complete. Model saved.")
