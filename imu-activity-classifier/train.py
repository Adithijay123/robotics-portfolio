"""
train.py
Trains a Random Forest and an SVM on UCI HAR IMU data.
Saves models and plots confusion matrices + feature importances.

Usage:  python train.py
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline

from features import load_raw_signals, load_labels, extract_features, LABELS

os.makedirs("./outputs", exist_ok=True)
os.makedirs("./models", exist_ok=True)

LABEL_NAMES = [LABELS[i] for i in range(1, 7)]


# ── 1. Load data ────────────────────────────────────────────────────────────

print("Loading raw sensor windows...")
X_train_raw = load_raw_signals("train")
X_test_raw  = load_raw_signals("test")

print("Extracting hand-crafted features (mean, std, RMS, FFT)...")
X_train = extract_features(X_train_raw)
X_test  = extract_features(X_test_raw)

y_train = load_labels("train") - 1   # 0-indexed
y_test  = load_labels("test")  - 1

print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"  Classes: {[LABELS[i+1] for i in range(6)]}\n")


# ── 2. Helpers ───────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_pct, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
        ax=ax, cbar_kws={"label": "% of true class"}
    )
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(f"./outputs/{filename}", dpi=150)
    plt.close()
    print(f"  Saved ./outputs/{filename}")


def evaluate(model, name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"  Test accuracy: {acc*100:.2f}%")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))
    return y_pred, acc


# ── 3. Random Forest ─────────────────────────────────────────────────────────

print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_train, y_train)

y_pred_rf, acc_rf = evaluate(rf, "Random Forest")
plot_confusion_matrix(y_test, y_pred_rf,
    f"Random Forest  —  {acc_rf*100:.1f}% accuracy",
    "confusion_rf.png")

# Feature importance plot
n_signals = 9
feature_labels = []
signal_names = [
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z",
    "tot_x", "tot_y", "tot_z",
]
stat_names = ["mean", "std", "min", "max", "rms", "fft_freq", "fft_peak"]
for s in signal_names:
    for st in stat_names:
        feature_labels.append(f"{s}_{st}")

importances = rf.feature_importances_
top_k = 20
top_idx = np.argsort(importances)[-top_k:][::-1]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(top_k), importances[top_idx], color="#4A90D9")
ax.set_xticks(range(top_k))
ax.set_xticklabels([feature_labels[i] for i in top_idx], rotation=45, ha="right", fontsize=8)
ax.set_title("Top 20 feature importances — Random Forest", fontsize=12)
ax.set_ylabel("Importance")
plt.tight_layout()
plt.savefig("./outputs/feature_importance.png", dpi=150)
plt.close()
print("  Saved ./outputs/feature_importance.png")


# ── 4. SVM ───────────────────────────────────────────────────────────────────

print("\nTraining SVM (RBF kernel) — this takes ~1–2 min on CPU...")
svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=10, gamma="scale", random_state=42)),
])
svm_pipe.fit(X_train, y_train)

y_pred_svm, acc_svm = evaluate(svm_pipe, "SVM (RBF kernel)")
plot_confusion_matrix(y_test, y_pred_svm,
    f"SVM (RBF)  —  {acc_svm*100:.1f}% accuracy",
    "confusion_svm.png")


# ── 5. Comparison bar chart ───────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(5, 4))
models = ["Random Forest", "SVM (RBF)"]
accs   = [acc_rf * 100, acc_svm * 100]
bars = ax.bar(models, accs, color=["#4A90D9", "#E07B39"], width=0.4)
for bar, a in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, a + 0.3, f"{a:.1f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylim(80, 100)
ax.set_ylabel("Test accuracy (%)")
ax.set_title("Model comparison — UCI HAR IMU dataset", fontsize=11)
plt.tight_layout()
plt.savefig("./outputs/model_comparison.png", dpi=150)
plt.close()
print("\n  Saved ./outputs/model_comparison.png")


# ── 6. Save models ────────────────────────────────────────────────────────────

with open("./models/random_forest.pkl", "wb") as f:
    pickle.dump(rf, f)
with open("./models/svm_pipeline.pkl", "wb") as f:
    pickle.dump(svm_pipe, f)

print("\nModels saved to ./models/")
print("\nAll done! Check ./outputs/ for plots.")
