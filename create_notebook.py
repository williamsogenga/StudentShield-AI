"""Create a readable, executable notebook that documents each assignment stage."""
import nbformat as nbf
from pathlib import Path
ROOT=Path(__file__).resolve().parent
nb=nbf.v4.new_notebook()
nb.cells=[
nbf.v4.new_markdown_cell("""# StudentShield AI — Complete ML Lab Notebook

**AI Mini-Project (10 assessment stages):** dropout risk and academic support screening. Source: [UCI #697](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) (CC BY 4.0). The target is a 3-class academic outcome (Dropout, Enrolled, Graduate).

> **Do not present synthetic demo scores as UCI findings.** To use real data run `python download_data.py` and `python train.py` first. This notebook contains the same logic but adds explanatory plots and tables. Prediction is at the **end of the second semester**, not enrollment. No Kenyan external validation has been done.

Start Jupyter in the **project root folder**, not inside the notebooks folder; alternatively let the next cell locate the project root."""),
nbf.v4.new_markdown_cell("## 1. Problem and goal\nUniversities may need better ways to prioritize voluntary student support. This project tests whether semester progress and tuition status can predict historical dropout/enrollment/graduation outcomes. This is a **multiclass supervised classification** task; it cannot independently decide which student will leave university."),
nbf.v4.new_code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
root = Path.cwd()
if not (root / 'core.py').exists(): root = root.parent
sys.path.insert(0, str(root))
from core import RAW_FEATURES, ALL_FEATURES, load_dataset, prepare_features
from sklearn.model_selection import train_test_split
real = root / 'data' / 'uci_students.csv'
demo = root / 'data' / 'demo_synthetic.csv'
path = real if real.exists() else demo
print('DATA SOURCE:', path.name, '  |  IS REAL UCI?', path == real)
if path == demo: print('WARNING: SYNTHETIC DEMO - NOT REAL DATA / NOT VALID ASSIGNMENT METRICS')
df = load_dataset(path)
print('Rows:',len(df),'Columns:',len(df.columns));df.head()"""),
nbf.v4.new_markdown_cell("## 2. Data acquisition\nOfficial UCI data #697 has 4,424 rows, 36 input variables and one three-class target. We use 11 input variables. The simulated demo file is explicitly labelled. Raw official data can be downloaded with `python download_data.py`."),
nbf.v4.new_code_cell("""print('Source:',path.name)
print('Targets:\\n',df['Target'].value_counts())
print('Available data types:\\n',df[RAW_FEATURES].dtypes.value_counts())"""),
nbf.v4.new_markdown_cell("## 3. Cleaning and preprocessing\nNormalize header whitespace/BOM, remove exact duplicate rows, coerce selected fields to numeric, and use a train-only median imputer inside each model pipeline. Random stratified 80:20 split avoids fitting scaling/imputation on test data."),
nbf.v4.new_code_cell("""print('Missing values:',df[RAW_FEATURES].isna().sum().sum())
print('Remaining exact duplicates:',df.duplicated().sum())
X = prepare_features(df)
y = df['Target']
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=.2,random_state=42)
print('Training:', X_train.shape, '| Test:',X_test.shape)"""),
nbf.v4.new_markdown_cell("## 4. Exploratory data analysis (EDA)\nVisualize class balance, grade distribution, and correlations; discuss sample patterns without causal claims."),
nbf.v4.new_code_cell("""fig, axes=plt.subplots(1,2,figsize=(12,4))
df['Target'].value_counts().plot.bar(ax=axes[0],title='Historical academic outcomes')
for t,part in df.groupby('Target'):
    axes[1].hist(part['Curricular units 2nd sem (grade)'].dropna(),bins=18,alpha=.55,label=t)
axes[1].legend();axes[1].set_title('Second-semester grade distribution')
plt.tight_layout();plt.show()"""),
nbf.v4.new_code_cell("""plt.figure(figsize=(10,6));sns.heatmap(df[RAW_FEATURES].corr(),cmap='BrBG',center=0)
plt.title('Selected numeric correlations');plt.tight_layout();plt.show()"""),
nbf.v4.new_markdown_cell("## 5. Feature engineering\nCreate approval-rate ratios for semesters 1 and 2, and grade change. Zero enrolled units yield approval rate 0 by explicit convention. The raw binary tuition/debtor/scholarship indicators are already encoded as 0/1. Logistic Regression uses scaling fitted on training folds."),
nbf.v4.new_code_cell("""X[['Sem 1 approval rate','Sem 2 approval rate','Grade change']].head(8)"""),
nbf.v4.new_markdown_cell("## 6. Model training\nRun `python train.py` for reproducible three-fold stratified CV with GridSearchCV over Logistic Regression and Random Forest, plus saved model/artifacts. Model selection uses **training CV macro-F1 only** (test set not used for tuning)."),
nbf.v4.new_code_cell("""from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold,GridSearchCV
cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
prep=ColumnTransformer([('numeric', Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),ALL_FEATURES)])
pipe=Pipeline([('prep',prep),('model',LogisticRegression(max_iter=1500,random_state=42))])
search=GridSearchCV(pipe, {'model__C':[.1,1,10]},cv=cv,scoring='f1_macro',n_jobs=-1)
search.fit(X_train,y_train)
print('Example training-fold CV macro-F1:',round(search.best_score_,3),'Best C:',search.best_params_)
print('For complete 2-model comparison, run python train.py')"""),
nbf.v4.new_markdown_cell("## 7. Held-out model evaluation\nMetrics are loaded from `reports/metrics.json` after running `python train.py`. Class-specific precision/recall/F1, macro-F1 and a confusion matrix are all included."),
nbf.v4.new_code_cell("""import json
j=root/'reports'/('metrics.json' if real.exists() else 'demo_metrics.json')
metrics=json.loads(j.read_text()) if j.exists() else None
if metrics:
    print('RESULT ORIGIN:', metrics['data_kind'])
    display(pd.DataFrame({n:{'accuracy':v['accuracy'],'f1_macro':v['f1_macro'],'cv_f1_macro':v['cv_f1_macro']} for n,v in metrics['results'].items()}).T)
    print('Model selected by CV:',metrics['best_model'])
else: print('Run python train.py to generate metrics')"""),
nbf.v4.new_markdown_cell("## 8. Interpretation\nPermutation importance describes what predictions rely on *in this dataset*; it is not an estimate of cause or of the impact of an intervention. The appropriate institutional response is supportive outreach, never an automated penalty."),
nbf.v4.new_code_cell("""if metrics:
    display(pd.Series(metrics['feature_importance'],name='Decrease in macro-F1').to_frame())"""),
nbf.v4.new_markdown_cell("## 9. Deployment\nStart app: `python -m streamlit run app.py`. App supports one-record entry, CSV upload/download, model statistics and safety guidance. Saved model joblib files support **offline inference after installation**."),
nbf.v4.new_markdown_cell("## 10. Documentation, presentation and caveats\nSee README and StudentShield_Report.docx / StudentShield_Presentation.pptx. Major caveats: real data historical and Portuguese, no Kenya validation, incomplete outcome status for Enrolled, no decision automation, no causal explanations, no production access controls. GitHub repository link must be added after the student publishes the code."),
]
nb.metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.10"}}
dest=ROOT/"notebooks"/"StudentShield_ML_Workflow.ipynb"
nbf.write(nb,dest)
print(dest)
