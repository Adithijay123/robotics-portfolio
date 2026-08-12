import kagglehub
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sounddevice as sd
import wavio
import time
from joblib import dump, load
import serial
import csv


# Download RAVDESS dataset with kagglehub (first run takeS time)
path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
print("Path to dataset files:", path)

# Emotion labels
int2emotion = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
AVAILABLE_EMOTIONS = {"angry", "sad", "neutral", "happy", "calm"}

def extract_feature(file, mfcc=True, chroma=True, mel=True):
    X, sample_rate = librosa.load(file, res_type="kaiser_fast")
    result = np.array([])
    if mfcc:
        mfccs = np.mean(
            librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0
        )
        result = np.hstack((result, mfccs))
    if chroma:
        stft = np.abs(librosa.stft(X))
        chroma_feat = np.mean(
            librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0
        )
        result = np.hstack((result, chroma_feat))
    if mel:
        mel_feat = np.mean(
            librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0
        )
        result = np.hstack((result, mel_feat))
    return result

def load_data(test_size=0.2):
    X, y = [], []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".wav"):
                basename = os.path.basename(file)
                try:
                    emotion = int2emotion[basename.split("-")[2]]
                except KeyError:
                    continue
                if emotion not in AVAILABLE_EMOTIONS:
                    continue
                features = extract_feature(os.path.join(root, file))
                X.append(features)
                y.append(emotion)
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)

# Train model only if not already saved
model_path = "emotion_model.joblib"
if not os.path.exists(model_path):
    print("Extracting features and training model (this will take a few minutes)...")
    X_train, X_test, y_train, y_test = load_data(test_size=0.2)
    print("Training classifier...")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    print("Model accuracy:", model.score(X_test, y_test))
    dump(model, model_path)  # Save model
    print("Model saved as emotion_model.joblib")
else:
    print("Loading trained model for fast prediction...")
    model = load(model_path)

def record_voice(duration=5, filename="recorded.wav"):
    print("\n--- New interaction ---")
    print("Recording starting in...")
    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)
    print("Recording now. Speak naturally near the cube.")
    fs = 44100
    rec = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wavio.write(filename, rec, fs, sampwidth=2)
    print("Done recording!")

# SERIAL TO PICO 
# match Serial.begin(115200)
ser = serial.Serial("COM28", 115200, timeout=1)
time.sleep(2)  # Let Pico reset after opening port

# map model emotion strings to Arduino letters
emotion_to_letter = {
    "happy": "H",
    "calm": "C",
    "sad": "S",
    "angry": "A",
    "neutral": "N",
}

def send_emotion_to_arduino(emotion_label: str, pattern_index: int):
    """
    Send emotion + pattern index over serial.
    For all emotions, we allow 0 or 1, but Arduino may ignore pattern for some.
    """
    letter = emotion_to_letter.get(emotion_label, "N")
    msg = f"{letter}{pattern_index}\n".encode("utf-8")
    ser.write(msg)
    print(f"Sent to Arduino: {msg.decode().strip()}")

# RL setup 

emotions = ["happy", "calm", "sad", "angry", "neutral"]

# Two patterns (0,1) for every emotion
patterns_per_emotion = {
    "happy": [0, 1],
    "calm": [0, 1],
    "sad": [0, 1],
    "angry": [0, 1],
    "neutral": [0, 1],
}

alpha = 0.1   # reward step up
beta = 0.1    # small decay when no feedback
epsilon_min = 0.05
epsilon_max = 0.95  # clamp maximum weight so no pattern dominates completely

# start all weights at 1.0
weights = {e: np.ones(len(patterns_per_emotion[e])) for e in emotions}

# interaction counter (no real timestamps logged for privacy)
interaction_id = 0

def select_pattern_for_emotion(e: str, eps: float = 0.1):
    """Epsilon-greedy pattern choice."""
    ws = weights[e]
    if np.random.rand() < eps:
        idx = np.random.randint(len(ws))
    else:
        idx = int(np.argmax(ws))
    pattern = patterns_per_emotion[e][idx]
    return idx, pattern

def update_weights(e: str, idx: int, feedback: int):
    """Update weights based on feedback: 1 or 0, with min/max clamp."""
    if e not in weights:
        return
    if feedback > 0:
        weights[e][idx] += alpha
    else:
        weights[e][idx] = max(weights[e][idx] - beta * 0.1, epsilon_min)

    # clamp all weights to [epsilon_min, epsilon_max]
    for j in range(len(weights[e])):
        weights[e][j] = min(max(weights[e][j], epsilon_min), epsilon_max)

    # normalise so they sum to 1
    s = weights[e].sum()
    if s > 0:
        weights[e] /= s

#  CSV logging 

log_path = "smushie_interactions.csv"

# create file with header if not present; otherwise append to existing file
if not os.path.exists(log_path):
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["interaction_id", "emotion_pattern", "feedback", "weights_after"])

def log_interaction(e: str, idx: int, feedback: int):
    global interaction_id
    interaction_id += 1

    # build label like happy_0, calm_1.
    emotion_pattern = f"{e}_{idx}"
    ws_after = ";".join([f"{w:.3f}" for w in weights[e]])

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([interaction_id, emotion_pattern, feedback, ws_after])

# Read feedback from Arduino

def read_feedback_window(window_seconds: float = 30.0) -> int:
    """
    Listen for FEEDBACK_POS from Arduino for a limited time.
    Return 1 if received, else 0.
    """
    ser.flushInput()
    end_time = time.time() + window_seconds
    feedback = 0
    print("Waiting for touch feedback (up to", window_seconds, "seconds)...")
    while time.time() < end_time:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        print("Serial:", line)
        if "FEEDBACK_POS" in line:
            feedback = 1
            break
    return feedback

# MAIN LOOP: continuous interactions so runs always after the first time

print("Smushie companion is now running. Press Ctrl+C to stop.")
try:
    while True:
        # 1) Record a short chunk of audio - auto saved as recorded.wav, not permenetly stored.
        record_voice(duration=5, filename="recorded.wav")

        # 2) Extract features and predict emotion
        features = extract_feature("recorded.wav")
        predicted_emotion = model.predict([features])
        pred = predicted_emotion[0]
        print("Raw predicted emotion:", pred)

        # 3) Only handle known emotions; otherwise fall back to neutral
        if pred not in emotions:
            pred = "neutral"
        print("Using emotion:", pred)

        # 4) Select pattern index (0 or 1 for every emotion)
        pattern_idx, pattern_id = select_pattern_for_emotion(pred, eps=0.1)
        print(f"Selected pattern index {pattern_idx} for {pred}")

        # 5) Send emotion + pattern index to  Pico
        send_emotion_to_arduino(pred, pattern_idx)

        # 6) Listen for touch feedback during the emotion window (up to 30 s)
        fb = read_feedback_window(window_seconds=30.0)
        print("Feedback value:", fb)

        # 7) Update weights and log
        update_weights(pred, pattern_idx, fb)
        log_interaction(pred, pattern_idx, fb)

        print("Updated weights for", pred, ":", weights[pred])
        print("Interaction logged to", log_path)

        # 8) Small pause before the next interaction
        time.sleep(30.0)

except KeyboardInterrupt:
    print("\nStopping Smushie companion. Goodbye!")
    ser.close()
