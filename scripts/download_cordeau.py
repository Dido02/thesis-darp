"""Download all Cordeau DARP instances and solution files.

Run from thesis-darp/:
    python scripts/download_cordeau.py
"""

import urllib.request
from pathlib import Path

BASE_URL = "http://neumann.hec.ca/chairedistributique/data/darp/tabu"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "cordeau"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INSTANCES = [f"pr{i:02d}" for i in range(1, 21)]

for name in INSTANCES:
    for suffix in ["", ".res"]:
        filename = f"{name}{suffix}"
        url = f"{BASE_URL}/{filename}"
        out_path = OUT_DIR / (f"{filename}.txt" if not suffix else filename)

        if out_path.exists():
            print(f"  skip {filename} (exists)")
            continue

        print(f"  downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, out_path)
        except Exception as e:
            print(f"  FAILED: {e}")

print(f"\nDone. Files in {OUT_DIR}")