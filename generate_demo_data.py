"""Generate clearly labelled SYNTHETIC data solely for an offline UI demonstration.
Run download_data.py and train.py for the actual UCI-based assignment evaluation.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from core import ROOT


def generate(n=1200, seed=41):
    r = np.random.default_rng(seed)
    age = np.clip(r.normal(23, 5, n).round(), 17, 62).astype(int)
    admission = np.clip(r.normal(132, 15, n), 85, 195).round(1)
    scholarship = r.binomial(1, .22, n)
    debtor = r.binomial(1, .16, n)
    fees = r.binomial(1, np.clip(.84 - .31*debtor, .2, .95))
    e1 = r.integers(4, 9, n)
    e2 = r.integers(4, 9, n)
    propensity = np.clip(.78 + (admission-130)/145 - .20*debtor + .07*scholarship + r.normal(0, .20, n), .10, .99)
    a1 = r.binomial(e1, propensity)
    a2 = r.binomial(e2, np.clip(propensity + r.normal(0, .12, n), .08, .99))
    g1 = np.where(a1 > 0, np.clip(8 + 8*a1/e1+r.normal(0, 2.2, n), 0, 20), 0).round(2)
    g2 = np.where(a2 > 0, np.clip(8 + 8*a2/e2+r.normal(0, 2.2, n), 0, 20), 0).round(2)
    score = -1.8 + 3.0*(1-a2/e2) + 1.7*(1-a1/e1) + .75*debtor + 1.15*(1-fees) + .04*(age-22) - .04*(g2-11)
    p_dropout = 1/(1+np.exp(-score))
    outcome = []
    for p in p_dropout:
        # Nontrivial stochastic outcomes: predictive rather than deterministic classification.
        p_enrolled = .17
        p_drop = float(p * (1-p_enrolled))
        outcome.append(r.choice(["Dropout", "Enrolled", "Graduate"], p=[p_drop, p_enrolled, 1-p_enrolled-p_drop]))
    data = pd.DataFrame({
        "Age at enrollment": age,
        "Admission grade": admission,
        "Tuition fees up to date": fees,
        "Debtor": debtor,
        "Scholarship holder": scholarship,
        "Curricular units 1st sem (enrolled)": e1,
        "Curricular units 1st sem (approved)": a1,
        "Curricular units 1st sem (grade)": g1,
        "Curricular units 2nd sem (enrolled)": e2,
        "Curricular units 2nd sem (approved)": a2,
        "Curricular units 2nd sem (grade)": g2,
        "Target": outcome,
    })
    return data

if __name__ == "__main__":
    dest = ROOT / "data" / "demo_synthetic.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    generate().to_csv(dest, index=False)
    print(f"Wrote {dest} (SYNTHETIC; NOT UCI DATA)")
