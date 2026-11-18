"""Reproducible ML pipeline: EDA, feature engineering, CV, evaluation, artifacts.
Usage: python download_data.py && python train.py
Optional clearly labelled synthetic smoke test: python train.py --demo
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from core import ROOT, RAW_FEATURES, ALL_FEATURES, TARGET_LABELS, load_dataset, prepare_features

sns.set_theme(style="whitegrid")


def save_charts(df: pd.DataFrame, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    order = [x for x in TARGET_LABELS if x in df.Target.unique()]
    ax = sns.countplot(x="Target", data=df, order=order, color="#147d87")
    for bar in ax.patches:
        ax.annotate(f"{int(bar.get_height()):,}", (bar.get_x()+bar.get_width()/2, bar.get_height()),
                    ha="center", va="bottom", fontsize=10)
    plt.title("Distribution of student outcomes")
    plt.tight_layout();plt.savefig(destination / "01_outcomes.png",dpi=170);plt.close()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    sns.boxplot(data=df, x="Target", y="Curricular units 2nd sem (grade)", order=order, ax=ax, color="#6da9a5")
    ax.set_title("Semester 2 grades by eventual outcome")
    plt.tight_layout();plt.savefig(destination / "02_grade_distribution.png",dpi=170);plt.close()
    corr_cols = ["Admission grade", "Age at enrollment", "Curricular units 1st sem (approved)",
                 "Curricular units 1st sem (grade)", "Curricular units 2nd sem (approved)",
                 "Curricular units 2nd sem (grade)"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap="BrBG",fmt=".2f",vmin=-1,vmax=1)
    plt.title("Numeric feature correlation (EDA)")
    plt.xticks(rotation=40, ha="right",fontsize=8);plt.yticks(fontsize=8)
    plt.tight_layout();plt.savefig(destination / "03_correlations.png",dpi=170);plt.close()


def main(demo: bool = False):
    dataset = ROOT / "data" / ("demo_synthetic.csv" if demo else "uci_students.csv")
    if not dataset.exists():
        raise SystemExit("Dataset missing. Run `python download_data.py` first, then `python train.py`."
                         " For a SYNTHETIC demo only: python generate_demo_data.py && python train.py --demo")
    df = load_dataset(dataset)
    # Source contains no IDs; exact-row duplicate removal is transparent and recorded.
    X = prepare_features(df)
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    prep_scaled = ColumnTransformer([("num",Pipeline([
        ("imputer", SimpleImputer(strategy="median")), ("scaler",StandardScaler())
    ]), ALL_FEATURES)], remainder="drop")
    prep_unscaled = ColumnTransformer([("num",SimpleImputer(strategy="median"),ALL_FEATURES)],remainder="drop")
    models = {
        "Logistic Regression": (Pipeline([("prep",prep_scaled),
                              ("model",LogisticRegression(max_iter=1500,random_state=42))]),
                              {"model__C":[.1,1,10]}),
        "Random Forest": (Pipeline([("prep",prep_unscaled),
                         ("model",RandomForestClassifier(random_state=42,n_jobs=1,class_weight="balanced_subsample"))]),
                         {"model__n_estimators":[120,240], "model__max_depth":[8,None], "model__min_samples_leaf":[2,5]})
    }
    results, fitted = {}, {}
    for name, (pipe, grid) in models.items():
        search = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=2,refit=True)
        search.fit(X_train,y_train)
        estimator = search.best_estimator_
        prediction = estimator.predict(X_test)
        p,r,f,s = precision_recall_fscore_support(y_test,prediction,labels=TARGET_LABELS,zero_division=0)
        results[name] = {
            "accuracy": round(float(accuracy_score(y_test,prediction)),4),
            "f1_macro":round(float(f1_score(y_test,prediction,average="macro")),4),
            "cv_f1_macro":round(float(search.best_score_),4),
            "best_params":search.best_params_,
            "per_class":{label:{"precision":round(float(p[i]),4),"recall":round(float(r[i]),4),
                                 "f1":round(float(f[i]),4),"support":int(s[i])} for i,label in enumerate(TARGET_LABELS)},
            "confusion_matrix":confusion_matrix(y_test,prediction,labels=TARGET_LABELS).tolist(),
        }
        fitted[name]=estimator
        print(f"{name}: holdout accuracy={results[name]['accuracy']:.3f}, macro F1={results[name]['f1_macro']:.3f}; CV macro F1={results[name]['cv_f1_macro']:.3f}")
    # Select model ONLY using training-fold cross-validation; never test metrics.
    best_name=max(results,key=lambda n: results[n]["cv_f1_macro"])
    selected=fitted[best_name]
    suffix="demo" if demo else "real"
    model_path=ROOT / "models" / ("demo_model.joblib" if demo else "student_model.joblib")
    model_path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(selected,model_path,compress=3)
    outdir=ROOT/"reports"/("demo_figures" if demo else "figures")
    save_charts(df,outdir)
    plt.figure(figsize=(5.5,4.5))
    sns.heatmap(results[best_name]["confusion_matrix"], annot=True,fmt="d",cmap="Blues",
                xticklabels=TARGET_LABELS,yticklabels=TARGET_LABELS)
    plt.xlabel("Predicted");plt.ylabel("Actual");plt.title(f"Held-out confusion matrix — {best_name}")
    plt.tight_layout();plt.savefig(outdir/"04_confusion_matrix.png",dpi=170);plt.close()
    # Permutation importance only descriptive, not used to choose features/models.
    imp=permutation_importance(selected,X_test,y_test,n_repeats=3,random_state=42,
                               scoring="f1_macro",n_jobs=2)
    importance=pd.Series(imp.importances_mean,index=ALL_FEATURES).sort_values(ascending=False)
    importance.to_csv(ROOT/"reports"/f"{suffix}_feature_importance.csv",header=["mean_delta_macro_f1"])
    plt.figure(figsize=(8,5))
    importance.head(10).sort_values().plot.barh(color="#147d87")
    plt.title("Permutation importance (held-out sample; descriptive)")
    plt.xlabel("Decrease in macro-F1 when permuted")
    plt.tight_layout();plt.savefig(outdir/"05_feature_importance.png",dpi=170);plt.close()
    summary={
        "data_kind":"SYNTHETIC DEMONSTRATION — NOT REAL UCI RESULTS" if demo else "UCI official dataset (real historical Portugal data)",
        "dataset_source":"synthetic internal generator" if demo else "UCI Machine Learning Repository, dataset #697 (CC BY 4.0)",
        "rows_after_deduplication":int(len(df)),
        "train_size":int(len(X_train)),"test_size":int(len(X_test)),
        "missing_values_raw":int(df[RAW_FEATURES].isna().sum().sum()),
        "class_counts":{k:int(v) for k,v in df.Target.value_counts().items()},
        "feature_count_raw":len(RAW_FEATURES),"feature_count_after_engineering":len(ALL_FEATURES),
        "best_model":best_name,"selection_rule":"highest 3-fold train CV macro-F1 (not test result)",
        "random_state":42,"split":"80% training / 20% untouched stratified test",
        "results":results,
        "feature_importance":{k:round(float(v),4) for k,v in importance.head(10).items()},
    }
    summary_path=ROOT/"reports"/("demo_metrics.json" if demo else "metrics.json")
    summary_path.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("Saved model:",model_path)
    print("Saved metrics:",summary_path)
    return summary

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--demo",action="store_true",help="Train on synthetic data; NOT valid real-world performance")
    args=p.parse_args()
    main(demo=args.demo)
