# Dataset provenance and conditions

**Intended real training dataset**: V. Realinho, M. Vieira Martins, J. Machado, and L. Baptista (2021), *Predict Students' Dropout and Academic Success*, UCI Machine Learning Repository, Dataset 697. DOI: https://doi.org/10.24432/C5MC89. Licence: Creative Commons Attribution 4.0 (CC BY 4.0). Dataset URL: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

Dataset facts: 4,424 individual records, 36 predictors, 1 target (Dropout, Enrolled, Graduate), from a Portuguese higher education institution. StudentShield selects 11 available features and derives three rates/differences. This is **not** a dataset about Kenyan students.

`uci_students.csv` is created by `python download_data.py` (requires internet). The installer checks 4,424 records and 37 columns before saving. Training with this file produces `models/student_model.joblib` and `reports/metrics.json`.

`demo_synthetic.csv`, if present, was **artificially generated** by `generate_demo_data.py` solely for offline demonstration of the software workflow; it must not be presented as real student data or real UCI model performance. The demo model and metrics have matching names (`demo_model.joblib`, `demo_metrics.json`).

No original raw personal identifiers are included. Do not upload real identifiable student records without suitable authorization. Do not use model predictions to deny scholarships/admission or discipline students.
