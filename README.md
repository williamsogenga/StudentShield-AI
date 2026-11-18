# StudentShield AI

**Student Dropout Risk and Early Support Dashboard** — standalone Python/scikit-learn and Streamlit educational project.

> **Important**: Experimental academic project, not a validated intervention or student-screening product. Dataset is from Portugal; not validated on Kenyan institutions. Human review is mandatory. Never use predictions to deny admission, scholarships or academic services.

## What the app does

- Supports a university academic adviser in reviewing **end-of-second-semester** academic indicators.
- Runs a genuine, persisted classification pipeline for **Dropout / Enrolled / Graduate**.
- Shows the model's predicted *Dropout* class probability with illustrative review-priority bands.
- Screens one de-identified record or a batch CSV, exports results, shows EDA and model evaluation.
- Gives simple rule-based student-support suggestions; these are not causal or personal feature explanations.

## Easy Windows installation (requires internet the first time)

1. Install Python 3.10 or newer (check **Add Python to PATH**).
2. Unzip this folder (never run scripts from inside the compressed ZIP).
3. Double-click **`setup_windows.bat`**. It creates `.venv`, installs dependencies, downloads official UCI data, trains the models, then starts Streamlit.
4. Open the shown local link, normally **http://localhost:8501**.
5. On later runs, double-click **`run_windows.bat`**.

If the Windows setup script does not recognize `py -3.10`, use this manual procedure:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python download_data.py
python train.py
python -m streamlit run app.py
```

`setup_windows.bat` tries `py -3.10` and otherwise `python`. To run the automated tests: `python tests.py`.

## Data and evaluation

Original source: V. Realinho et al. (2021), [Predict Students' Dropout and Academic Success](https://doi.org/10.24432/C5MC89), [UCI Repository #697](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success), licence **CC BY 4.0**. There are 4,424 records, 36 predictors and three academic outcomes in the original. `download_data.py` fetches `data.csv` and validates its shape before saving `data/uci_students.csv`.

- Select **11** input variables, focusing on semester marks/approvals and tuition-related context; calculate **3** derived numeric features.
- Strip inconsistent header whitespace, coerce numeric fields, handle missing data using a **train-only median imputer** inside each pipeline, drop exact duplicate records, and stratify the 80:20 train/test split.
- Compare Logistic Regression and Random Forest using **3-fold stratified CV + grid search** on training data only. Select the highest CV macro-F1 before evaluating on the untouched test sample.
- Calculate held-out accuracy, macro-F1, per-class precision/recall/F1, confusion matrix, and descriptive permutation importance.
- Save a trained model (`models/student_model.joblib`), JSON metrics (`reports/metrics.json`), charts (`reports/figures`) and importance CSV.

### Offline demonstration option (synthetic, not valid academic results)

If unable to download UCI data, you can still **check the app interface** using `python generate_demo_data.py`, then `python train.py --demo`, then `python -m streamlit run app.py`. The dashboard displays a highly visible **synthetic demo** warning. Neither `reports/demo_metrics.json` nor the demo model is evidence of real-world performance. **Replace demo with the official UCI training** before reporting results to your lecturer.

## File structure

```
StudentShield_AI_Project/
├── app.py                   # Streamlit dashboard / predictions
├── core.py                  # Feature engineering, human support rules
├── train.py                 # Models, CV, held-out evaluation, EDA
├── download_data.py         # Authenticated UCI dataset download
├── generate_demo_data.py    # Synthetic UI testing only
├── tests.py                 # Automated tests
├── data/                    # UCI raw csv after download; demo CSV
├── models/                  # Saved model(s) after training
├── reports/                 # Evaluation metrics and EDA charts
├── notebooks/               # Walkthrough notebook
├── sample_upload.csv        # In data/; example batch input
├── StudentShield_Report.docx
├── StudentShield_Presentation.pptx
├── setup_windows.bat / run_windows.bat
└── requirements.txt
```

## Batch upload CSV

The 11 required headers are listed in the app and `data/sample_upload.csv`; use only anonymized rows. Do not upload names or student IDs. Risk bands (0.40, 0.70) are **illustrative**, not validated thresholds. The Streamlit app does not implement logins, databases, or institutional security controls; use it locally with authorized demo data.

## Reproducibility & academic honesty

`python train.py` rebuilds all official-data model artifacts (random seed 42). If you change source data, report regenerated metrics instead of copying older numbers. The source dataset is attributed; synthetic model metrics are named `demo_...` and labelled accordingly. Source repository: https://github.com/williamsogenga/StudentShield-AI. Generated models and datasets are excluded from version control and rebuilt by the setup scripts. The committed Word report and PowerPoint are **synthetic demonstration templates**; after training with the official UCI dataset, regenerate both documents using `python make_submission.py` and commit the real-data versions before final submission.

## Presentation walkthrough

Launch the app; navigate **Overview → Assess a student → Batch screening → Model & insights → About and safeguards**. Explain the difference between predictors and target, approval-rate feature engineering, CV versus held-out evaluation, the confusion matrix and ethical safeguards. Allow a lecturer to enter different records to see how probabilities change.
