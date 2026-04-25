"""
explore.py
Quick EDA — plot raw sensor traces and per-class feature distributions.
Run after download_data.py:  python explore.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from features import load_raw_signals, load_labels, LABELS

LABEL_NAMES = [LABELS[i] for i in range(1, 7)]
COLORS = ["#4A90D9", "#E07B39", "#5AB88A", "#A86BD6", "#D9574A", "#7FBFBF"]

import os
os.makedirs("./outputs", exist_ok=True)

print("Loading data...")
X_raw = load_raw_signals("train")   # (7352, 128, 9)
y     = load_labels("train") - 1   # 0-indexed

# ── Plot 1: Sample raw signal traces for each activity ────────────────────

fig = plt.figure(figsize=(14, 9))
fig.suptitle("Raw accelerometer X — one window per activity", fontsize=13)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.3)

for act_idx in range(6):
    ax = fig.add_subplot(gs[act_idx // 3, act_idx % 3])
    sample_idx = np.where(y == act_idx)[0][0]
    trace = X_raw[sample_idx, :, 0]   # body_acc_x
    ax.plot(trace, color=COLORS[act_idx], linewidth=1.2)
    ax.set_title(LABEL_NAMES[act_idx], fontsize=10)
    ax.set_xlabel("Sample (50 Hz)", fontsize=8)
    ax.set_ylabel("Body acc X (g)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_ylim(-2.5, 2.5)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

plt.savefig("./outputs/raw_traces.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved ./outputs/raw_traces.png")


# ── Plot 2: Class distribution ────────────────────────────────────────────

counts = [(LABEL_NAMES[i], np.sum(y == i)) for i in range(6)]
labels, vals = zip(*counts)

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(labels, vals, color=COLORS)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 20, str(v),
            ha="center", va="bottom", fontsize=9)
ax.set_title("Training samples per activity class", fontsize=12)
ax.set_ylabel("Count")
plt.xticks(rotation=25, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig("./outputs/class_distribution.png", dpi=150)
plt.close()
print("Saved ./outputs/class_distribution.png")


# ── Plot 3: Per-class mean + std band for acc_x ──────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
t = np.arange(128)
for act_idx in range(6):
    mask    = y == act_idx
    windows = X_raw[mask, :, 0]           # (n, 128)
    mu      = windows.mean(axis=0)
    sigma   = windows.std(axis=0)
    ax.plot(t, mu, label=LABEL_NAMES[act_idx], color=COLORS[act_idx], linewidth=1.5)
    ax.fill_between(t, mu - sigma, mu + sigma, color=COLORS[act_idx], alpha=0.12)

ax.set_title("Per-class mean ± std — body acc X (train set)", fontsize=12)
ax.set_xlabel("Sample (50 Hz → 2.56 s window)")
ax.set_ylabel("Acceleration (g)")
ax.legend(fontsize=8, loc="upper right")
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
plt.tight_layout()
plt.savefig("./outputs/mean_traces.png", dpi=150)
plt.close()
print("Saved ./outputs/mean_traces.png")

print("\nEDA complete. Check ./outputs/ for plots.")
