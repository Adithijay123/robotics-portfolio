"""
download_data.py
Downloads the UCI HAR dataset and saves it to ./data/
Run this first: python download_data.py

If the download fails (the UCI repo sometimes blocks scripts),
manually download from:
  https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
and unzip it into ./data/ so the folder structure is:
  ./data/UCI HAR Dataset/train/...
  ./data/UCI HAR Dataset/test/...
"""

import urllib.request
import zipfile
import os

# UCI repo sometimes requires browser-like headers
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)
DATA_DIR = "./data"
ZIP_PATH = "./data/uci_har.zip"


def download():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists("./data/UCI HAR Dataset"):
        print("Dataset already downloaded.")
        return

    print("Downloading UCI HAR Dataset (~60 MB)...")
    req = urllib.request.Request(DATA_URL, headers={
        "User-Agent": "Mozilla/5.0"
    })
    with urllib.request.urlopen(req) as response:
        with open(ZIP_PATH, "wb") as f:
            f.write(response.read())

    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)
    os.remove(ZIP_PATH)

    # The zip may unpack as 'UCI HAR Dataset' directly or inside another folder
    # Normalise the path
    for candidate in ["UCI HAR Dataset", "human activity recognition using smartphones"]:
        p = os.path.join(DATA_DIR, candidate)
        if os.path.exists(p) and candidate != "UCI HAR Dataset":
            os.rename(p, os.path.join(DATA_DIR, "UCI HAR Dataset"))
            break

    print("Done. Dataset saved to ./data/UCI HAR Dataset/")


if __name__ == "__main__":
    download()
