"""StudentShield AI: local Streamlit educational early-support prototype.
Run: streamlit run app.py
Never use for automated disciplinary, admissions or scholarship decisions.
"""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from core import ROOT, RAW_FEATURES, ALL_FEATURES, prepare_features, load_dataset, risk_band, support_suggestions

st.set_page_config(page_title="StudentShield AI | Student Success Intelligence",page_icon="🎓",layout="wide",initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');
html, body, [class*="css"], [data-testid="stApp"] {font-family:'DM Sans',sans-serif;}
[data-testid="stAppViewContainer"]{background:linear-gradient(135deg,#f6f9fa 0%,#fff 65%);}
[data-testid="stSidebar"]{background:#0c2434;color:white;}
[data-testid="stSidebar"] * {color:#f4f9f9 !important;}
h1,h2,h3{font-family:'Outfit',sans-serif;letter-spacing:-.025em;}
h1{color:#0f2c3a;font-size:2.2rem !important;}h2{color:#113747;}
[data-testid="stMetric"]{background:white;padding:17px 20px;border:1px solid #e3ebee;border-radius:14px;box-shadow:0 2px 12px #0e344009;}
[data-testid="stMetricLabel"]{font-size:.88rem;color:#55707e;}
[data-testid="stMetricValue"]{font-family:'Outfit',sans-serif;color:#0e4350;}
.hero {background:linear-gradient(115deg,#0f3341 0%,#155e6b 72%,#1a898b 100%);padding:32px 36px;border-radius:20px;color:#fff;margin-bottom:24px;}
.hero h1{color:#fff!important;margin:0 0 8px!important;font-size:2.3rem!important}
.hero p{opacity:.92;font-size:1.04rem;max-width:820px;margin:0;line-height:1.65}
.kicker{color:#83e5d6;font-size:.75rem;letter-spacing:.14em;font-weight:800;margin-bottom:8px}
.stButton>button[kind="primary"]{background:#147b81;border-color:#147b81;border-radius:9px}
.muted{color:#526c77;font-size:.91rem}
</style>
""",unsafe_allow_html=True)

@st.cache_resource
def get_assets():
    real_model=ROOT/"models"/"student_model.joblib"
    demo_model=ROOT/"models"/"demo_model.joblib"
    if real_model.exists() and (ROOT/"reports"/"metrics.json").exists():
        model=real_model; metrics=ROOT/"reports"/"metrics.json"; demo=False
    elif demo_model.exists() and (ROOT/"reports"/"demo_metrics.json").exists():
        model=demo_model; metrics=ROOT/"reports"/"demo_metrics.json"; demo=True
    else:
        return None,None,None
    return joblib.load(model),json.loads(metrics.read_text(encoding="utf-8")),demo

model,summary,demo=get_assets()
with st.sidebar:
    st.markdown("## 🎓 STUDENTSHIELD")
    st.caption("ACADEMIC SUPPORT INTELLIGENCE")
    st.divider()
    section=st.radio("Navigation",["Overview","Assess a student","Batch screening","Model & insights","About and safeguards"],label_visibility="collapsed")
    st.divider()
    if summary:
        st.caption("MODEL STATUS")
        st.success("Offline model loaded")
        st.caption("Synthetic demonstration" if demo else "Authentic UCI model")
        st.caption(f"Selected: {summary['best_model']}")
    else:
        st.error("Model not trained")
    st.markdown("<span style='font-size:.76rem'>For academic demonstration, not autonomous student decisions.</span>",unsafe_allow_html=True)

if model is None:
    st.error("No trained model found. Run `python download_data.py`, then `python train.py`, then restart this application. For a synthetic UI-only demo, run `python generate_demo_data.py` and `python train.py --demo`.")
    st.stop()
if demo:
    st.warning("**SYNTHETIC DEMO MODE** — The current model was trained on generated sample records, not the official UCI data. Predictions and performance are for interface testing only. Run `python download_data.py` and `python train.py` before academic evaluation.")
else:
    st.info("Research demonstration: model trained on historical Portuguese university records. It has NOT been validated for Kenyan universities or any individual student.")

figures=ROOT/"reports"/("demo_figures" if demo else "figures")
data_path=ROOT/"data"/("demo_synthetic.csv" if demo else "uci_students.csv")


def human_probs(frame: pd.DataFrame):
    probs=model.predict_proba(prepare_features(frame))
    classes=list(model.classes_)
    return pd.DataFrame(probs,columns=classes,index=frame.index)


def hero(tag,heading,subtitle):
    st.markdown(f'<div class="hero"><div class="kicker">{tag}</div><h1>{heading}</h1><p>{subtitle}</p></div>',unsafe_allow_html=True)

if section=="Overview":
    hero("STUDENT SUCCESS / OVERVIEW","Support students before they disengage","An advisor-first dashboard for understanding historical learning patterns and screening records that merit a supportive human conversation.")
    counts=summary.get("class_counts",{})
    cols=st.columns(4)
    cols[0].metric("Historical records",f"{summary['rows_after_deduplication']:,}")
    cols[1].metric("Historical dropouts",f"{counts.get('Dropout',0):,}")
    cols[2].metric("Held-out macro F1",f"{summary['results'][summary['best_model']]['f1_macro']:.3f}")
    cols[3].metric("Features used",f"{summary['feature_count_after_engineering']}")
    st.write("")
    left,right=st.columns([1.1,1])
    with left:
        st.subheader("Student outcome distribution")
        if (figures/"01_outcomes.png").exists():st.image(str(figures/"01_outcomes.png"),use_container_width=True)
        st.caption("Counts describe the source data, not current enrollment rates.")
    with right:
        st.subheader("How screening works")
        st.markdown("""**1 · Enter** a de-identified end-of-second-semester student record.

**2 · Estimate** three outcome probabilities with a saved scikit-learn classifier.

**3 · Review** the dropout probability alongside other outcomes and simple support suggestions.

**4 · Act responsibly** — an academic adviser confirms context before making any support referral.""")
        st.info("The priority bands (40% and 70%) are illustrative UI thresholds, **not clinically or institutionally validated cutoffs**.")

elif section=="Assess a student":
    hero("SCREENING / INDIVIDUAL","Review one student record","Supply academic indicators from the first two semesters. No student name or ID is required or stored.")
    defaults={"Age at enrollment":21,"Admission grade":132.0,"Tuition fees up to date":1,"Debtor":0,"Scholarship holder":0,
              "Curricular units 1st sem (enrolled)":6,"Curricular units 1st sem (approved)":5,"Curricular units 1st sem (grade)":13.5,
              "Curricular units 2nd sem (enrolled)":6,"Curricular units 2nd sem (approved)":5,"Curricular units 2nd sem (grade)":13.0}
    with st.form("single_assessment"):
        a,b=st.columns(2)
        with a:
            st.markdown("#### Enrollment and support indicators")
            age=st.number_input("Age at enrollment",min_value=17,max_value=100,value=defaults["Age at enrollment"])
            grade=st.number_input("Admission grade (0–200)",min_value=0.,max_value=200.,value=132.,step=1.)
            fees=st.selectbox("Tuition fees up to date?",["Yes","No"])
            debtor=st.selectbox("Outstanding debt indicator?",["No","Yes"])
            scholarship=st.selectbox("Scholarship holder?",["No","Yes"])
        with b:
            st.markdown("#### Academic progress")
            e1=st.number_input("Semester 1: units enrolled",min_value=0,max_value=30,value=6)
            a1=st.number_input("Semester 1: units approved",min_value=0,max_value=30,value=5)
            g1=st.number_input("Semester 1: mean grade (0–20)",min_value=0.,max_value=20.,value=13.5,step=.5)
            e2=st.number_input("Semester 2: units enrolled",min_value=0,max_value=30,value=6)
            a2=st.number_input("Semester 2: units approved",min_value=0,max_value=30,value=5)
            g2=st.number_input("Semester 2: mean grade (0–20)",min_value=0.,max_value=20.,value=13.,step=.5)
        go=st.form_submit_button("Calculate review priority",type="primary",use_container_width=True)
    if go:
        if a1>e1 or a2>e2:
            st.error("Approved units cannot exceed enrolled units; correct the counts and retry.")
        else:
            record={"Age at enrollment":age,"Admission grade":grade,"Tuition fees up to date":int(fees=="Yes"),
                    "Debtor":int(debtor=="Yes"),"Scholarship holder":int(scholarship=="Yes"),
                    "Curricular units 1st sem (enrolled)":e1,"Curricular units 1st sem (approved)":a1,
                    "Curricular units 1st sem (grade)":g1,"Curricular units 2nd sem (enrolled)":e2,
                    "Curricular units 2nd sem (approved)":a2,"Curricular units 2nd sem (grade)":g2}
            ps=human_probs(pd.DataFrame([record])).iloc[0]
            risk=float(ps["Dropout"])
            st.subheader("Screening result")
            x,y=st.columns([.42,.58])
            with x:
                st.metric("Estimated dropout class probability",f"{risk:.1%}")
                st.markdown(f"**Review category:** {risk_band(risk)}")
                st.caption("Probability is a model estimate, not an observed student's future.")
            with y:
                st.bar_chart(ps.rename("Probability"),horizontal=True)
            st.markdown("#### Suggested supportive follow-up")
            for suggestion in support_suggestions(record):
                st.markdown("• "+suggestion)
            st.caption("Suggestions are transparent simple rules based on entered fields, not SHAP explanations or evidence of causality.")

elif section=="Batch screening":
    hero("SCREENING / BATCH","Screen a de-identified CSV","Upload multiple anonymous student records, see probability-based review priorities, and download the results.")
    sample_path=ROOT/"data"/"sample_upload.csv"
    if sample_path.exists():
        st.download_button("Download example CSV template",sample_path.read_bytes(),file_name="students_example.csv",mime="text/csv")
    up=st.file_uploader("Choose a CSV with the required columns",type=["csv"])
    with st.expander("Show 11 required column names"):
        st.code("\n".join(RAW_FEATURES),language="text")
    if up:
        try:
            raw=pd.read_csv(up)
            if "Target" in raw: raw=raw.drop(columns=["Target"])
            clean=raw.copy()
            engineered=prepare_features(clean)
            if len(raw)>10000:raise ValueError("Upload no more than 10,000 rows at once")
            prob=human_probs(clean)
            result=pd.DataFrame({"Dropout probability":prob["Dropout"],
                                 "Enrolled probability":prob["Enrolled"],
                                 "Graduate probability":prob["Graduate"]})
            result["Review priority"]=result["Dropout probability"].map(risk_band)
            if "Student reference" in raw.columns:
                result.insert(0,"Student reference",raw["Student reference"].astype(str).values)
            result=result.sort_values("Dropout probability",ascending=False).reset_index(drop=True)
            c1,c2,c3=st.columns(3)
            c1.metric("Rows processed",len(result))
            c2.metric("High review priority",int((result["Review priority"]=="High review priority").sum()))
            c3.metric("Average model dropout probability",f"{result['Dropout probability'].mean():.1%}")
            st.dataframe(result,use_container_width=True,hide_index=True)
            st.download_button("Download assessment results",result.to_csv(index=False).encode(),
                               "students_screening_results.csv","text/csv",type="primary")
            st.caption("Uploaded data are processed in the running app session only; the code does not save uploads to disk. Use anonymous records for demonstrations.")
        except Exception as exc:
            st.error(f"Unable to screen CSV: {exc}")

elif section=="Model & insights":
    hero("TRANSPARENCY / MODEL","Model performance & evidence","See how the models were compared and what the untouched test partition can and cannot demonstrate.")
    rows=[]
    for name,vals in summary["results"].items():
        rows.append({"Model":name,"CV macro F1 (training only)":vals["cv_f1_macro"],
                     "Test macro F1":vals["f1_macro"],"Test accuracy":vals["accuracy"]})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    st.caption(f"Selection: {summary['selection_rule']}. Stratified 80:20 test split. Random seed {summary['random_state']}.")
    c1,c2=st.columns(2)
    with c1:
        st.subheader("Confusion matrix")
        if (figures/"04_confusion_matrix.png").exists():st.image(str(figures/"04_confusion_matrix.png"),use_container_width=True)
    with c2:
        st.subheader("Features most useful to prediction")
        if (figures/"05_feature_importance.png").exists():st.image(str(figures/"05_feature_importance.png"),use_container_width=True)
        st.caption("Permutation importance measures predictive utility, not causal impact.")
    with st.expander("Explore more EDA figures"):
        for fig,title in [("02_grade_distribution.png","Grades by outcome"),("03_correlations.png","Correlation matrix")]:
            st.subheader(title)
            if (figures/fig).exists():st.image(str(figures/fig),use_container_width=True)
    st.markdown("#### Per-class performance")
    cls=summary["results"][summary["best_model"]]["per_class"]
    st.dataframe(pd.DataFrame(cls).T,use_container_width=True)

elif section=="About and safeguards":
    hero("GOVERNANCE / PURPOSE","Designed to support, not label","The interface explores model-assisted academic advising in a controlled teaching demonstration.")
    st.markdown("""### Appropriate use
An academic adviser may use results to **offer** optional tutoring, fee-related help or a welfare conversation. A person must review each case and consult the student.

### Important limits
- **External validity**: the underlying dataset comes from Portugal; it is not validated for Kenya or any other university.
- **Time of prediction**: both semester 1 and semester 2 results are inputs. The system cannot screen brand-new entrants.
- **Outcome ambiguity**: *Enrolled* is an intermediate historical status, not guaranteed long-term success.
- **Uncalibrated probabilities**: numbers are model scores, not validated individual dropout frequencies.
- **Potential unfairness**: patterns can reflect socioeconomic circumstances or historic inequity; the score should not trigger sanctions.
- **Privacy**: never upload identifiable student data without explicit permission and an institutional legal basis. No database or upload logging is implemented.

### Data & acknowledgement
V. Realinho, M. Vieira Martins, J. Machado, and L. Baptista (2021), **Predict Students' Dropout and Academic Success**, UCI ML Repository #697 (CC BY 4.0), DOI: https://doi.org/10.24432/C5MC89.

### Reproducibility
`download_data.py` obtains the real CSV; `train.py` performs the split, three-fold cross-validation, model comparison, charts and model export. The Jupyter notebook documents the same workflow.
""")
