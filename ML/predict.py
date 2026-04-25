"""
predict.py
Load a saved model and predict the activity from a single sensor window.

Usage:
    python predict.py --window sample_window.csv --model rf
    python predict.py --window sample_window.csv --model svm

The CSV should have 9 columns (body_acc_x/y/z, body_gyro_x/y/z, total_acc_x/y/z)
and exactly 128 rows (one 2.56-second window at 50 Hz).

To generate a sample window for testing, run:
    python predict.py --generate-sample
"""

import argparse
import pickle
import numpy as np
import pandas as pd

from features import extract_features, LABELS

MODEL_PATHS = {
    "rf":  "./models/random_forest.pkl",
    "svm": "./models/svm_pipeline.pkl",
}

LABEL_NAMES = [LABELS[i] for i in range(1, 7)]


def load_model(name: str):
    path = MODEL_PATHS[name]
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_window(csv_path: str, model_name: str = "rf"):
    model = load_model(model_name)

    df = pd.read_csv(csv_path, header=None)
    assert df.shape == (128, 9), (
        f"Expected (128, 9) window, got {df.shape}. "
        "File must have 128 rows and 9 sensor columns."
    )

    window = df.values.astype(np.float32)               # (128, 9)
    features = extract_features(window[np.newaxis])     # (1, n_features)

    pred_idx  = model.predict(features)[0]
    proba     = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
    elif hasattr(model, "named_steps"):
        clf = model.named_steps.get("svm")
        if hasattr(clf, "predict_proba"):
            proba = model.predict_proba(features)[0]

    print(f"\nModel:      {model_name.upper()}")
    print(f"Prediction: {LABEL_NAMES[pred_idx]}")
    if proba is not None:
        print("\nClass probabilities:")
        for name, p in sorted(zip(LABEL_NAMES, proba), key=lambda x: -x[1]):
            bar = "█" * int(p * 30)
            print(f"  {name:<22} {p*100:5.1f}%  {bar}")


def generate_sample(activity_idx: int = 0):
    """Generate a synthetic window for quick testing."""
    np.random.seed(42)
    t = np.linspace(0, 2.56, 128)
    window = np.zeros((128, 9))
    # Simulate walking: periodic acc_x + noise
    window[:, 0] = 0.8 * np.sin(2 * np.pi * 1.8 * t) + 0.1 * np.random.randn(128)
    window[:, 1] = 0.4 * np.sin(2 * np.pi * 1.8 * t + 0.5) + 0.1 * np.random.randn(128)
    window[:, 2] = 9.8 + 0.2 * np.random.randn(128)
    for ch in range(3, 9):
        window[:, ch] = 0.1 * np.random.randn(128)

    pd.DataFrame(window).to_csv("sample_window.csv", header=False, index=False)
    print("Generated sample_window.csv (simulated walking signal)")
    print("Run:  python predict.py --window sample_window.csv --model rf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=str, help="Path to 128×9 CSV window")
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "svm"])
    parser.add_argument("--generate-sample", action="store_true")
    args = parser.parse_args()

    if args.generate_sample:
        generate_sample()
    elif args.window:
        predict_window(args.window, args.model)
    else:
        parser.print_help()
