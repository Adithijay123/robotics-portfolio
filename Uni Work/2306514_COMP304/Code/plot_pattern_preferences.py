import csv
import numpy as np
import matplotlib.pyplot as plt

log_path = "smushie_interactions.csv"

emotions = ["happy", "calm", "sad", "angry", "neutral"]
patterns_per_emotion = {
    "happy":   [0, 1],
    "calm":    [0, 1],
    "sad":     [0, 1],
    "angry":   [0, 1],
    "neutral": [0, 1],
}

def load_latest_weights():
    """
    Read the CSV with header:
    timestamp,emotion,pattern_index,feedback,weights_after
    and return latest weights per emotion. If weights are missing,
    fall back to normalised selection counts.
    """
    counts = {e: np.zeros(len(patterns_per_emotion[e]), dtype=int) for e in emotions}
    last_weights = {e: None for e in emotions}

    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            e = row["emotion"]             
            # "happy" or "happy_0" in some rows
            if "_" in e:
            # handle older rows like "happy_0"
                e, _ = e.split("_", 1)
            if e not in emotions:
                continue

            try:
                idx = int(row["pattern_index"])
            except ValueError:
            # if pattern_index is bad, skip
                continue

            if 0 <= idx < len(counts[e]):
                counts[e][idx] += 1

            weights_after = row.get("weights_after", "")
            if weights_after:
                last_weights[e] = weights_after

    parsed_weights = {}
    for e in emotions:
        if last_weights[e] is not None and ";" in last_weights[e]:
            vals = [float(x) for x in last_weights[e].split(";")]
            parsed_weights[e] = np.array(vals)
        else:
            total = counts[e].sum()
            if total > 0:
                parsed_weights[e] = counts[e] / total
            else:
                parsed_weights[e] = np.ones(len(patterns_per_emotion[e])) / len(patterns_per_emotion[e])

    return parsed_weights

def plot_preferences():
    weights = load_latest_weights()

    num_emotions = len(emotions)
    fig, axes = plt.subplots(1, num_emotions, figsize=(3 * num_emotions, 3), sharey=True)
    if num_emotions == 1:
        axes = [axes]

    for ax, e in zip(axes, emotions):
        w = weights[e]
        pattern_labels = [f"{e}_{i}" for i in range(len(w))]
        ax.bar(pattern_labels, w, color=["#A4C0EE", "#C183C9"])
        ax.set_title(e.capitalize())
        ax.set_ylim(0, 1)
        ax.set_xticklabels(pattern_labels, rotation=45, ha="right", fontsize=8)

    fig.suptitle("Current pattern-selection probabilities per emotion", fontsize=12)
    fig.tight_layout()
    plt.savefig("pattern_preferences.png", dpi=200)
    # plt.show()  # optional

if __name__ == "__main__":
    plot_preferences()
    print("Saved plot to pattern_preferences.png")
