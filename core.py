"""Shared, transparent feature preparation and guidance for StudentShield AI."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_FEATURES = [
    "Age at enrollment",
    "Admission grade",
    "Tuition fees up to date",
    "Debtor",
    "Scholarship holder",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
]
ENGINEERED_FEATURES = ["Sem 1 approval rate", "Sem 2 approval rate", "Grade change"]
ALL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES
TARGET_LABELS = ["Dropout", "Enrolled", "Graduate"]


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip BOM, line breaks, tabs and duplicate spaces in column names."""
    frame = frame.copy()
    frame.columns = [" ".join(str(c).replace("\ufeff", " ").split()) for c in frame.columns]
    return frame


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = normalize_columns(pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig"))
    missing = [name for name in RAW_FEATURES + ["Target"] if name not in df]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    df = df.drop_duplicates().copy()
    for name in RAW_FEATURES:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    df["Target"] = df["Target"].astype(str).str.strip().str.title()
    df = df[df["Target"].isin(TARGET_LABELS)].copy()
    if len(df) < 100 or df["Target"].nunique() != 3:
        raise ValueError("Expected >=100 records with Dropout, Enrolled and Graduate classes")
    return df


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(frame)
    missing = [name for name in RAW_FEATURES if name not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    features = df[RAW_FEATURES].copy()
    for col in RAW_FEATURES:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    # Defensive handling of clearly impossible rates. NaNs are handled inside
    # the train-fitted imputer, not using the held-out test data.
    for sem in ("1st", "2nd"):
        enrolled = features[f"Curricular units {sem} sem (enrolled)"].clip(lower=0)
        approved = features[f"Curricular units {sem} sem (approved)"].clip(lower=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = approved / enrolled.replace(0, np.nan)
        features[f"Sem {'1' if sem == '1st' else '2'} approval rate"] = rate.clip(0, 1).fillna(0)
    features["Grade change"] = (
        features["Curricular units 2nd sem (grade)"] - features["Curricular units 1st sem (grade)"]
    )
    return features[ALL_FEATURES]


def risk_band(dropout_probability: float) -> str:
    # Illustrative and unvalidated thresholds, not institutional policy.
    if dropout_probability >= 0.70:
        return "High review priority"
    if dropout_probability >= 0.40:
        return "Moderate review priority"
    return "Routine monitoring"


def support_suggestions(record: dict) -> list[str]:
    """Simple human-readable rules, NOT learned causal explanations."""
    suggestions = []
    for sem in ("1st", "2nd"):
        attempted = float(record.get(f"Curricular units {sem} sem (enrolled)", 0) or 0)
        passed = float(record.get(f"Curricular units {sem} sem (approved)", 0) or 0)
        if attempted > 0 and passed / attempted < 0.6:
            suggestions.append("Offer academic advising and optional tutoring for low course completion.")
            break
    if float(record.get("Tuition fees up to date", 1) or 0) == 0 or float(record.get("Debtor", 0) or 0) == 1:
        suggestions.append("Offer a confidential conversation about bursaries or payment-plan options.")
    if float(record.get("Curricular units 2nd sem (grade)", 0) or 0) < 10:
        suggestions.append("Offer study-support resources and review the student's assessment difficulties.")
    if not suggestions:
        suggestions.append("Maintain routine, supportive academic check-ins.")
    suggestions.append("Confirm any concerns directly with the student; do not make decisions from the score alone.")
    return list(dict.fromkeys(suggestions))
