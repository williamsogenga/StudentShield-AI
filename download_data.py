"""Download the official UCI #697 dataset; must be run with internet access.
Original data is licensed under CC BY 4.0; attribution in data/SOURCE.md.
"""
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import requests
import pandas as pd
from core import ROOT

URLS = [
    "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip",
    "https://raw.githubusercontent.com/ASIF-Kh/Student-Dropout-Prediction/main/data.csv",
]


def download_dataset():
    dest = ROOT / "data" / "uci_students.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in URLS:
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "StudentShield-Education-Project/1.0"})
            r.raise_for_status()
            payload = r.content
            if payload[:2] == b"PK":
                with ZipFile(BytesIO(payload)) as archive:
                    candidates = [n for n in archive.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX")]
                    if not candidates:
                        raise ValueError("UCI archive has no CSV")
                    payload = archive.read(next((n for n in candidates if n.lower().endswith("data.csv")), candidates[0]))
            raw = pd.read_csv(BytesIO(payload), sep=";", encoding="utf-8-sig")
            if len(raw) != 4424 or "Target" not in raw or raw.shape[1] != 37:
                raise ValueError(f"Unexpected file shape {raw.shape}; expected (4424, 37)")
            dest.write_bytes(payload)
            print(f"SUCCESS: {len(raw)} authentic UCI records -> {dest}")
            print(f"Source: {url}")
            return dest
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Download unsuccessful. You can manually save UCI data.csv as data/uci_students.csv.\n" + "\n".join(errors))

if __name__ == "__main__":
    download_dataset()
