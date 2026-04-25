# IMU Gesture / Activity Classifier

A CPU-only machine learning project that classifies human activities (walking, sitting, standing, etc.) from raw IMU (accelerometer + gyroscope) sensor data.

Built for a robotics portfolio — demonstrates the full ML pipeline from raw sensor windows to trained model inference.

---

## Dataset

**UCI HAR Dataset** — 30 subjects wearing a smartphone on their waist performing 6 activities.
- 50 Hz sampling rate, 2.56 s sliding windows (128 samples)
- 9 sensor channels: body acc (x/y/z), gyroscope (x/y/z), total acc (x/y/z)
- 7,352 training windows / 2,947 test windows

---

## Results

| Model         | Test Accuracy |
|---------------|---------------|
| Random Forest | ~92–94%       |
| SVM (RBF)     | ~93–95%       |

Outputs saved to `./outputs/`:
- `raw_traces.png` — sample signal per activity
- `mean_traces.png` — per-class mean ± std band
- `class_distribution.png` — training set balance
- `confusion_rf.png` — Random Forest confusion matrix
- `confusion_svm.png` — SVM confusion matrix
- `feature_importance.png` — top 20 RF features
- `model_comparison.png` — side-by-side accuracy

---

## Setup

### Requirements
```
Python 3.9+
pandas
scikit-learn
matplotlib
seaborn
numpy
```

Install all at once:
```bash
pip install pandas scikit-learn matplotlib seaborn numpy
```

---

## Usage

### 1. Download the dataset
```bash
python download_data.py
```

### 2. Explore the raw signals (optional but recommended)
```bash
python explore.py
```

### 3. Train both models
```bash
python train.py
```
Training takes ~2–3 minutes on a laptop CPU.

### 4. Run inference on a new window
```bash
# Generate a synthetic sample window
python predict.py --generate-sample

# Predict with Random Forest
python predict.py --window sample_window.csv --model rf

# Predict with SVM
python predict.py --window sample_window.csv --model svm
```

---

## Project Structure

```
imu_gesture_classifier/
├── download_data.py   # Download UCI HAR dataset
├── features.py        # Raw signal loading + feature extraction
├── explore.py         # EDA plots
├── train.py           # Train RF + SVM, save models + plots
├── predict.py         # Inference on a single window CSV
├── data/              # Dataset (created by download_data.py)
├── models/            # Saved .pkl models (created by train.py)
└── outputs/           # All plots (created by explore.py / train.py)
```

---

## Feature Engineering

From each 128-sample window, 7 statistics are extracted per channel (9 channels × 7 = 63 features total):

| Feature | Description |
|---------|-------------|
| Mean | Central tendency |
| Std | Signal variability |
| Min / Max | Range |
| RMS | Signal energy |
| FFT peak freq | Dominant frequency (normalised) |
| FFT peak magnitude | Strength of dominant frequency |

---

## Why this matters for robotics

- IMU sensors are on every robot, drone, and wearable
- Sliding window + feature extraction is the standard pipeline for embedded inference
- Random Forest is interpretable — the feature importance plot shows *which* signals and statistics matter most
- The predict.py script mirrors how you'd deploy this: take a raw sensor buffer, extract features, classify

---

## Extensions (if you have more time)

- Replace the feature vector with an LSTM on raw windows (add `torch`)
- Deploy to a Raspberry Pi and stream live phone IMU data via OSC
- Add a ROS2 node that subscribes to `/imu` and publishes predicted activity
- Implement a sliding window buffer for real-time inference
