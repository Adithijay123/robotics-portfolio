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
import random

# EMOTION MODEL PART

# Download RAVDESS dataset (first run only; then cached)
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
    "08": "surprised"
}
AVAILABLE_EMOTIONS = {"angry", "sad", "neutral", "happy"}


def extract_feature(file, mfcc=True, chroma=True, mel=True):
    X, sample_rate = librosa.load(file, res_type='kaiser_fast')
    result = np.array([])
    if mfcc:
        mfccs = np.mean(
            librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,
            axis=0
        )
        result = np.hstack((result, mfccs))
    if chroma:
        stft = np.abs(librosa.stft(X))
        chroma_feat = np.mean(
            librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,
            axis=0
        )
        result = np.hstack((result, chroma_feat))
    if mel:
        mel_feat = np.mean(
            librosa.feature.melspectrogram(y=X, sr=sample_rate).T,
            axis=0
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


# Train or load model
model_path = "emotion_model.joblib"
if not os.path.exists(model_path):
    print("Extracting features and training model (this will take a few minutes)...")
    X_train, X_test, y_train, y_test = load_data(test_size=0.2)
    print("Training classifier...")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    print("Model accuracy:", model.score(X_test, y_test))
    dump(model, model_path)
    print("Model saved as emotion_model.joblib")
else:
    print("Loading trained model for fast prediction...")
    model = load(model_path)

# RECORDING PART

def record_voice(duration=5, filename="recorded.wav"):
    print("Recording starting in...")
    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)
    print("Recording now! Speak into the microphone.")
    fs = 44100
    rec = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wavio.write(filename, rec, fs, sampwidth=2)
    print("Done recording! Saved to", filename)

# ARDUINO + PATTERN LOGIC
PATTERN_COUNTS = {
    "happy": 3,
    "calm": 3,     # treat calm as happy-like for now
    "neutral": 2,
    "sad": 2,
    "angry": 2
}

PREF_FILE = "preferences.npy"

def load_preferences():
    if os.path.exists(PREF_FILE):
        return np.load(PREF_FILE, allow_pickle=True).item()

    prefs = {}
    for emo, n in PATTERN_COUNTS.items():
        w = np.ones(n, dtype=float)
        w /= w.sum()
        prefs[emo] = w
    np.save(PREF_FILE, prefs)
    return prefs

def save_preferences(prefs):
    np.save(PREF_FILE, prefs)

def choose_pattern_for_emotion(emotion, prefs):
    weights = prefs[emotion]
    indices = list(range(len(weights)))
    choice = random.choices(indices, weights=weights, k=1)[0]
    return choice

def update_preference(emotion, chosen_index, prefs, alpha=0.2):
    w = prefs[emotion].astype(float)
    w[chosen_index] += alpha
    w = np.maximum(w, 0.001)
    w /= w.sum()
    prefs[emotion] = w
    return prefs

def send_pattern_to_arduino(ser, emotion, pattern_index):
    emo_code = {
        "happy": "H",
        "calm": "C",
        "neutral": "N",
        "sad": "S",
        "angry": "A"
    }.get(emotion, "N")

    cmd = f"{emo_code}{pattern_index}\n"
    print("Sending to Arduino:", cmd.strip())
    ser.write(cmd.encode("utf-8"))

# Open serial to Arduino (change COM port if needed)
ser = serial.Serial('COM6', 9600, timeout=1)
time.sleep(2)  # Let Arduino reset


if __name__ == "__main__":
    # Record voice
    record_voice(duration=5, filename="recorded.wav")

    # Extract features and predict emotion
    features = extract_feature("recorded.wav")
    predicted_emotion = model.predict([features])[0]
    print("Predicted Emotion:", predicted_emotion)

    # Map emotion to known set (fallback to neutral if unknown)
    preferences = load_preferences()
    emotion_key = predicted_emotion if predicted_emotion in PATTERN_COUNTS else "neutral"

    # Choose LED pattern for this emotion
    pattern_idx = choose_pattern_for_emotion(emotion_key, preferences)
    print(f"Selected pattern {pattern_idx} for emotion {emotion_key}")

    # Send command to Arduino
    send_pattern_to_arduino(ser, emotion_key, pattern_idx)

    # Update preferences so this pattern becomes slightly more likely
    preferences = update_preference(emotion_key, pattern_idx, preferences, alpha=0.2)
    save_preferences(preferences)
    print("Updated preferences for", emotion_key, ":", preferences[emotion_key])
