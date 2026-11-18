"""Generate assignment-aligned DOCX and PPTX with CURRENT saved evaluation.
After real-data training: python make_submission.py
If only a synthetic demo model exists, both artifacts clearly disclose that fact.
"""
from __future__ import annotations
import json
from datetime import date
from pathlib import Path
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_BREAK
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from pptx import Presentation
from pptx.util import Inches as PI, Pt as PP
from pptx.dml.color import RGBColor as PRGB
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from core import ROOT, RAW_FEATURES, ALL_FEATURES

real=ROOT/"reports"/"metrics.json"
metrics_path=real if real.exists() else ROOT/"reports"/"demo_metrics.json"
if not metrics_path.exists():raise SystemExit("Train the model with python train.py (or --demo) first")
metrics=json.loads(metrics_path.read_text(encoding="utf-8"))
demo=metrics_path.name.startswith("demo_")
figure_dir=ROOT/"reports"/("demo_figures" if demo else "figures")
selected=metrics["best_model"]
result=metrics["results"][selected]
MODE_LABEL="SYNTHETIC DEMONSTRATION — NOT ACTUAL UCI RESULTS" if demo else "AUTHENTIC UCI DATASET — HISTORICAL PORTUGUESE RECORDS"
SOURCE_URL="https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success"

# ------------------- DOCUMENT -------------------
doc=Document()
sec=doc.sections[0]
sec.top_margin=Inches(.75);sec.bottom_margin=Inches(.75)
sec.left_margin=Inches(.90);sec.right_margin=Inches(.90)
styles=doc.styles
norm=styles["Normal"]
norm.font.name="Aptos";norm.font.size=Pt(10);norm.font.color.rgb=RGBColor(33,53,64)
norm.paragraph_format.space_after=Pt(5)
for name,sz,clr in [("Title",27,RGBColor(17,52,66)),("Heading 1",15,RGBColor(15,85,98)),("Heading 2",11,RGBColor(22,91,100))]:
    st=styles[name];st.font.name="Aptos Display";st.font.bold=True;st.font.size=Pt(sz);st.font.color.rgb=clr
    st.paragraph_format.space_before=Pt(10);st.paragraph_format.space_after=Pt(5)

def para(text="",style=None,boldlead=None):
    p=doc.add_paragraph(style=style)
    if boldlead and text.startswith(boldlead):
        p.add_run(boldlead).bold=True;p.add_run(text[len(boldlead):])
    else:p.add_run(text)
    return p

def bullet(text):
    return doc.add_paragraph(text,style="List Bullet")

def heading(text,level=1):doc.add_heading(text,level=level)

def table(headers,rows,widths=None):
    t=doc.add_table(rows=1,cols=len(headers));t.style="Light Shading Accent 1"
    t.alignment=WD_TABLE_ALIGNMENT.CENTER
    for i,h in enumerate(headers):
        c=t.rows[0].cells[i];c.text=str(h)
        tcPr=c._tc.get_or_add_tcPr();shd=OxmlElement('w:shd');shd.set(qn('w:fill'),'116574');tcPr.append(shd)
        for rr in c.paragraphs[0].runs:rr.font.bold=True;rr.font.color.rgb=RGBColor(255,255,255);rr.font.size=Pt(9)
    for row in rows:
        cells=t.add_row().cells
        for j,value in enumerate(row):
            cells[j].text=str(value)
            cells[j].vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in cells[j].paragraphs:
                p.paragraph_format.space_after=Pt(0)
                for r in p.runs:r.font.size=Pt(8.5)
    return t

def fig(name,caption,w=5.9):
    f=figure_dir/name
    if f.exists():
        p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.add_run().add_picture(str(f),width=Inches(w))
        q=doc.add_paragraph(caption);q.alignment=WD_ALIGN_PARAGRAPH.CENTER
        for rr in q.runs:rr.italic=True;rr.font.size=Pt(8);rr.font.color.rgb=RGBColor(73,95,108)

# Document header/footer
header=sec.header.paragraphs[0];header.text="STUDENTSHIELD AI    /    AI MINI-PROJECT";header.style="Caption"
footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
footer.add_run("StudentShield AI  •  Academic demonstration  •  Page ")
fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');footer._p.append(fld)

p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.space_before=Pt(65)
run=p.add_run("STUDENTSHIELD AI");run.bold=True;run.font.size=Pt(28);run.font.color.rgb=RGBColor(20,93,107)
p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=p.add_run("AI-Powered Student Dropout Risk\nand Early Support System");r.font.size=Pt(18);r.bold=True
p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.add_run("AI MINI-PROJECT REPORT  |  10-SECTION ASSIGNMENT").bold=True
p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.add_run("Prepared for academic assessment — 18 November 2026")
para("")
p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.add_run("Student name: __________________________________________")
p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.add_run("Registration number: __________________________________")
p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.add_run("Institution / course: ____________________________________")
para("")
p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;r=p.add_run(MODE_LABEL);r.bold=True;r.font.color.rgb=RGBColor(177,72,38) if demo else RGBColor(13,107,94)
para("Independent student academic project. This application is a teaching demonstration, not a validated university decision system.")
doc.add_page_break()

heading("Executive summary")
para("StudentShield AI is a local, adviser-facing machine learning prototype intended to show how historical first-year academic progress and financial-support indicators can help identify records for optional, supportive follow-up. Unlike an attendance tracker, it is a three-class prediction exercise (Dropout, Enrolled, Graduate). A second-semester snapshot is required; screening at admission would be misleading. Predictions are not decisions or guarantees.")
para("Real intended data source: Realinho et al. (2021), UCI #697, containing 4,424 Portuguese university records, 36 predictors and one target. The prototype selects 11 recorded predictors and engineers 3 numeric features. Logistic Regression and Random Forest are compared using training-only stratified cross-validation, followed by a held-out evaluation.")
if demo:para("CURRENT ARTIFACT STATUS: The data download was not available in the preparation environment. This report's charts and metrics are from explicitly SYNTHETIC DEMO DATA. Run the supplied download_data.py, train.py and make_submission.py on an internet-connected computer to generate the authentic-data evaluation. Synthetic metrics must not be submitted or described as actual UCI results.")
else:para(f"Authentic-data experimental result: the CV-selected {selected} achieved accuracy {result['accuracy']:.3f} and macro-F1 {result['f1_macro']:.3f} on the untouched test split (n={metrics['test_size']}). Generalizability to Kenya has not been established.")
heading("Assignment coverage")
table(["Assessment section", "Evidence within project"],[
("1. Problem definition","Section 1 / classification objective"),
("2. Data acquisition","Section 2 / UCI provenance"),
("3. Data cleaning","Section 3 / core.py and train.py"),
("4. EDA","Section 4 / figures + notebook"),
("5. Feature engineering","Section 5 / three derived features"),
("6. Model building","Section 6 / cross-validation + GridSearchCV"),
("7. Model evaluation","Section 7 / comparison, class metrics, confusion"),
("8. Interpretation","Section 8 / permutation feature importance"),
("9. Deployment","Section 9 / local Streamlit app"),
("10. Documentation & presentation","Section 10 / README, slides, GitHub to add")])
heading("1. Problem definition and objectives")
para("Problem. Universities face difficulties noticing academic disengagement early enough to offer appropriate support. Historical semester results and administrative indicators can contain useful predictive patterns, but reviewing every record manually can be inefficient. StudentShield AI tests whether such patterns predict recorded academic outcomes and displays them to human advisers without automating adverse action.")
para("Goal. Build and evaluate a reproducible multiclass classification system that displays the estimated probability of historical dropout, continued enrollment or graduation from available end-of-second-semester student features.")
bullet("Train and compare two supervised learning methods using transparent cross-validation.")
bullet("Provide individual and bulk CSV screening with exported predictions and optional support suggestions.")
bullet("Evaluate class-specific precision, recall, F1-score, accuracy and confusion matrices; document methodological and ethical limitations.")
heading("2. Data acquisition")
para("The official, publicly available UCI repository dataset (ID 697) was collected from a Portuguese higher-education institution and released by Realinho and colleagues. It contains 4,424 rows, 36 features and three categorical outcomes. We do not collect data from Kenyan students. CSV is downloaded with download_data.py, checked for exactly 4,424 records and 37 columns, and stored under data/uci_students.csv.")
para("Selected inputs are age at enrollment, admission grade, tuition-up-to-date flag, debtor flag, scholarship flag, and enrolled/approved/mean-grade values for each of semesters one and two. This reduced set avoids directly using gender, nationality and parental occupation and allows a comprehensible interface. It does NOT guarantee absence of unfairness.")
para("Dataset licence: CC BY 4.0; full attribution and DOI are in data/SOURCE.md. There are no student names or IDs in the project dataset.")
if demo:para("Evidence status: current figures use only data/demo_synthetic.csv (a generated simulation, not the UCI source). Authentic data download and rerun are required before presenting empirical conclusions.")
heading("3. Data cleaning and preprocessing")
for t in ["Read CSV with delimiter inference and UTF-8 BOM tolerance; normalize column whitespace and tab characters.",
          "Drop exact duplicated records; require valid outcomes; coerce numeric predictors, converting invalid text to missing values.",
          "Split outcome-stratified 80% training / 20% test with random_state=42 BEFORE fitting imputers or scalers.",
          "Fit median imputation only on training folds; standardize continuous inputs for Logistic Regression using a pipeline; Random Forest uses a non-scaled imputation pipeline.",
          "Do not remove whole records solely for extreme grades without a data-dictionary justification; constrain single-entry app inputs to plausible allowable ranges."]:
    bullet(t)
para(f"Processed rows (current mode): {metrics['rows_after_deduplication']:,}; training n={metrics['train_size']:,}; test n={metrics['test_size']:,}; missing values in selected raw fields: {metrics['missing_values_raw']}.")
heading("4. Exploratory data analysis (EDA)")
para("Class counts, grade distribution and correlation analysis characterize sample imbalance and associations. The plots are descriptive, not causal. A full reproducible notebook is included.")
fig("01_outcomes.png","Figure 1. Distribution across the three labelled outcomes ("+("synthetic sample" if demo else "UCI sample")+").",5.1)
fig("02_grade_distribution.png","Figure 2. Second-semester grade distribution by final outcome; association is not causation.",5.1)
fig("03_correlations.png","Figure 3. Selected numerical feature correlations (exploratory).",5.1)

heading("5. Feature engineering")
para("Three continuous variables are generated from the 11 selected raw inputs. Approval rate for each semester is 'units approved / units enrolled' (zero if enrolled = 0 by documented convention). Grade change is 'semester 2 average grade − semester 1 average grade'. These summarize academic progression while preserving raw inputs. Binary financial-support indicators remain 0/1. This yields 14 model features.")
table(["Derived feature","Calculation","Why included"],[
("Sem 1 approval rate","approved_1 / enrolled_1","Completion relative to workload"),
("Sem 2 approval rate","approved_2 / enrolled_2","Updated academic progression"),
("Grade change","mean_grade_2 − mean_grade_1","Improvement or decline")])
para("Feature engineering is a deterministic transformation inside core.py and is identically applied to training, test and uploaded records. All predictive inputs are observed no later than the second semester; the target is never included in features.")
heading("6. Model building and hyperparameter tuning")
para("Two supervised classifiers were trained on the same stratified training partition. Logistic Regression is an interpretable linear baseline; Random Forest is a nonlinear ensemble able to capture feature interactions. Three-fold StratifiedKFold with shuffling and seed 42 supports GridSearchCV. Macro-F1 is selected because the three outcome classes are imbalanced.")
table(["Classifier","Tuned parameters","Pipeline"],[
("Logistic Regression","C ∈ {0.1, 1, 10}; max_iter=1500","Median imputer + StandardScaler + model"),
("Random Forest","n_estimators {120, 240}; max_depth {8, None}; min_samples_leaf {2, 5}","Median imputer + classifier; balanced_subsample")])
para(f"Model chosen by training CV macro-F1: {selected}. The untouched test split was not used for model selection. Saved joblib model: models/{'demo_model' if demo else 'student_model'}.joblib.")
heading("7. Model evaluation")
para("The withheld 20% evaluation sample produces accuracy, macro-averaged F1, and per-class precision/recall/F1; the confusion matrix shows errors by actual and predicted class. Outcomes are Dropout, Enrolled and Graduate. Reported results are calculated, not predetermined.")
table(["Model","CV macro F1","Test accuracy","Test macro F1"],[
(n, f"{v['cv_f1_macro']:.3f}",f"{v['accuracy']:.3f}",f"{v['f1_macro']:.3f}") for n,v in metrics["results"].items()])
para(f"Selected model {selected} — held-out precision / recall / F1 by class:")
table(["Outcome","Precision","Recall","F1","N"],[
(c, f"{s['precision']:.3f}",f"{s['recall']:.3f}",f"{s['f1']:.3f}",str(s['support']))
 for c,s in result["per_class"].items()])
fig("04_confusion_matrix.png","Figure 4. Confusion matrix on untouched test sample; actual rows and predicted columns.",5.1)
if demo:para("ACADEMIC INTEGRITY NOTE: All numbers in Section 7 are from SYNTHETIC sample records and validate the software only. They are NOT findings from the UCI university dataset.")
heading("8. Interpretation, insights and recommendations")
para("Permutation importance tests how much held-out macro-F1 falls after each feature is randomly permuted; it is a measure of association for this fitted model, not a causal explanation. Top features in the current sample:")
table(["Feature","Permutation Δ macro F1"],[(k,f"{v:.3f}") for k,v in list(metrics['feature_importance'].items())[:7]])
fig("05_feature_importance.png","Figure 5. Descriptive permutation importance for the selected model.",5.0)
para("Operational recommendation: advisers may offer voluntary tutoring or academic advising for low semester approvals, or confidential information about bursary/payment support when financial flags are present. These are transparent rules based on entered data, not treatment-effect estimates. A high score is not proof of a problem, and a low score is not a reason to refuse help.")
heading("9. Deployment and demonstration")
para("The system is a Streamlit web interface running locally. It loads a saved scikit-learn estimator from joblib and offers five pages: Overview, Assess a student, Batch screening, Model & insights, and About and safeguards. Users enter 11 fields or upload a CSV, receive estimated class probabilities and review-priority bands, and can download batch results as CSV. The app does not store uploaded records in its own database.")
para("Reproduction (Windows): extract project; double-click setup_windows.bat with internet enabled to create a venv, install dependencies, download the UCI data, train models, regenerate report/slides and launch Streamlit. On subsequent runs start run_windows.bat, or use `python -m streamlit run app.py`. Local default is http://localhost:8501. Upload example: data/sample_upload.csv.")
para("Deployment scope is local proof of concept only; there is no authentication, student record integration, production-grade security or institution-specific validation. Avoid real identifiable student data.")
heading("10. Documentation, presentation and testing")
para("Included deliverables: source scripts; README; reproducible notebook; data provenance file; model and charts; automated unit tests; presentation slides; and this report. tests.py verifies feature dimensions, rate bounds, missing-column validation, risk bands and support-rule behavior. Publish the entire source folder to an individual GitHub repository without .venv, then insert the **real** GitHub link in the final submission.")
para("GitHub URL (to be completed after publication): ________________________________________________")
heading("Limitations and research ethics")
for t in [
"Portuguese historical data do not establish applicability to Kenyan universities or today's learners; no external validation was performed.",
"The input fields include second-semester grades; this is not a truly pre-entry early-warning prediction.",
"The Enrolled class is an intermediate status, and predictive class probabilities have not been formally calibrated.",
"Socioeconomic proxies may reproduce historical bias. Priority thresholds (40% / 70%) are UI demonstrations, not approved institutional policy.",
"Do not use outputs for penalties, admission denial, bursary exclusion, or other autonomous high-stakes decisions.",
"Future work: obtain consented institutional data, check calibration and group fairness, use time-aware validation, pilot review with real advisers, and secure any deployment."]:
    bullet(t)
heading("References")
para("[1] Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). Predict Students' Dropout and Academic Success [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89 (CC BY 4.0).")
para("[2] Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.")
para("[3] Streamlit documentation. https://docs.streamlit.io/ (accessed November 2026).")
para("[4] AI Mini – Group Project: lecturer's assignment brief (10 assessed criteria, 10 marks each).")
report_file=ROOT/"StudentShield_Report.docx"
doc.save(report_file)

# ------------------- SLIDES -------------------
prs=Presentation();prs.slide_width=PI(13.333);prs.slide_height=PI(7.5)
NAVY=PRGB(13,43,58); TEAL=PRGB(21,122,131); SOFT=PRGB(242,248,248); DARK=PRGB(37,64,75)
GREY=PRGB(101,122,132); WHITE=PRGB(255,255,255); GOLD=PRGB(252,184,89); RED=PRGB(177,75,48)

def rectangle(sl,x,y,w,h,color,radius=False):
    shape=sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,PI(x),PI(y),PI(w),PI(h))
    shape.fill.solid();shape.fill.fore_color.rgb=color;shape.line.fill.background()
    return shape

def tbox(sl,text,x,y,w,h,size=18,color=DARK,bold=False,align=None):
    s=sl.shapes.add_textbox(PI(x),PI(y),PI(w),PI(h));tf=s.text_frame;tf.word_wrap=True
    tf.margin_left=PI(.03);tf.margin_top=PI(.03);tf.margin_right=PI(.02)
    for i,line in enumerate(text.split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line
        p.font.name="Aptos Display" if bold else "Aptos";p.font.size=PP(size);p.font.bold=bold;p.font.color.rgb=color
        p.space_after=PP(9)
        if align:p.alignment=align
    return s

def base(title,sub=None,num=0):
    sl=prs.slides.add_slide(prs.slide_layouts[6]);rectangle(sl,0,0,13.333,7.5,WHITE)
    rectangle(sl,0,0,.16,7.5,TEAL)
    tbox(sl,"STUDENTSHIELD / AI MINI-PROJECT",.5,.22,6,.28,10,TEAL,True)
    tbox(sl,title,.55,.73,12, .85,31,NAVY,True)
    if sub:tbox(sl,sub,.59,1.55,11.7,.6,13,GREY)
    rectangle(sl,.57,7.12,12.2,.015,PRGB(225,234,236))
    tbox(sl,"StudentShield AI  •  Academic demonstration",.63,7.16,9,.18,9,GREY)
    tbox(sl,f"{num:02d} / 12",11.8,7.15,1,.22,9,GREY,True)
    if demo:
        rectangle(sl,9.1,.20,3.7,.28,PRGB(251,236,226),True)
        tbox(sl,"SYNTHETIC DEMO — NOT UCI RESULTS",9.17,.235,3.6,.21,9,RED,True)
    return sl

def add_image(sl,filename,x,y,w,h):
    path=figure_dir/filename
    if path.exists():
        from PIL import Image
        with Image.open(path) as im:iw,ih=im.size
        scale=min(w/iw,h/ih)
        nw,nh=iw*scale,ih*scale
        sl.shapes.add_picture(str(path),PI(x+(w-nw)/2),PI(y+(h-nh)/2),width=PI(nw),height=PI(nh))

def bigpoint(sl,n,head,desc,y):
    rectangle(sl,.78,y,.5,.5,TEAL,True)
    tbox(sl,str(n),.88,y+.04,.25,.4,17,WHITE,True)
    tbox(sl,head,1.5,y-.03,9.9,.45,19,NAVY,True)
    tbox(sl,desc,1.53,y+.42,10.7,.72,14,DARK)

# Cover
s=prs.slides.add_slide(prs.slide_layouts[6]);rectangle(s,0,0,13.333,7.5,NAVY)
rectangle(s,9.8,0,3.53,7.5,TEAL);rectangle(s,11.35,0,1.98,7.5,PRGB(21,134,144))
tbox(s,"AI MINI-PROJECT / PROJECT REPORT",.8,1.20,7,.36,14,PRGB(135,218,215),True)
tbox(s,"StudentShield\nAI",.78,1.87,7.9,2.05,47,WHITE,True)
tbox(s,"Dropout risk estimation and proactive\nstudent support",.86,4.25,7.4,1.25,23,PRGB(226,245,244))
tbox(s,"Prepared for academic assessment  •  18 Nov 2026",.87,6.55,8,.38,13,WHITE)
if demo:tbox(s,"SYNTHETIC DEMO\nReal-data training pending",9.83,5.4,3.2,.9,17,WHITE,True)
# 2
s=base("01  The problem", "Learning-support teams need a clearer way to prioritize voluntary academic outreach.",2)
bigpoint(s,1,"Manual reviews take time","Semester performance and administrative indicators may contain warning patterns.",2.32)
bigpoint(s,2,"The AI task","Supervised three-class prediction: Dropout, Enrolled, or Graduate.",3.61)
bigpoint(s,3,"Support, never sanction","Advisers use the result to start a conversation, not make an automated decision.",4.9)
# 3
s=base("02  Objective & system workflow","A repeatable, adviser-first analysis from historical data to a local app.",3)
steps=[("INPUT","11 fields\nTwo semesters"),("PREPARE","Clean + derive\n3 features"),("LEARN","Train + tune\n2 classifiers"),("REVIEW","3 outcomes +\nsupport options")]
for i,(a,b) in enumerate(steps):
    x=.7+i*3.15;rectangle(s,x,2.3,2.85,2.25,SOFT,True);rectangle(s,x,2.3,2.85,.12,TEAL)
    tbox(s,a,x+.18,2.72,2.5,.38,15,TEAL,True);tbox(s,b,x+.18,3.3,2.5,1.03,20,NAVY,True)
tbox(s,"Three-class probabilities, optional review categories, individual entry and batch CSV screening.",1,5.2,11.5,.9,22,DARK)
# 4
s=base("03  Data acquisition","Official source: UCI ML Repository #697; academic outcomes from Portugal.",4)
for i,(k,v) in enumerate([("4,424","Source records"),("36","Original predictor fields"),("3","Outcome classes")]):
    x=.7+i*4.12;rectangle(s,x,2.15,3.6,1.55,SOFT,True);tbox(s,k,x+.27,2.35,2.9,.67,37,TEAL,True);tbox(s,v,x+.29,3.13,3,.37,15,DARK)
tbox(s,"Data at enrollment and at the end of the first and second semesters. We use 11 selected predictors.\nLicense: CC BY 4.0. No Kenyan student data used or claimed.",.81,4.2,11.95,1.5,20,DARK)
# 5
s=base("04–05  Cleaning and features","Test leakage protection is part of the training pipeline.",5)
for i,(h,b) in enumerate([("CLEAN","Normalize headings • remove duplicates\nconvert numeric types • handle missing"),
                        ("SPLIT","80% train / 20% test, stratified\nseed = 42, untouched holdout"),
                        ("ENGINEER","Approval rates for two semesters\nchange in grade across semesters")]):
    x=.74+4.23*i;rectangle(s,x,2.35,3.88,2.65,SOFT,True);tbox(s,h,x+.23,2.67,3.5,.4,19,TEAL,True);tbox(s,b,x+.22,3.29,3.5,1.4,18,DARK)
tbox(s,"Imputation/scaling fitted inside training folds only — never globally before splitting.",.94,5.54,11.8,.78,19,NAVY,True)
# 6
s=base("04  EDA: sample outcomes","Class balance matters; macro-F1 evaluates each of three classes equally.",6)
add_image(s,"01_outcomes.png",.62,2.06,6.3,4.42);add_image(s,"02_grade_distribution.png",6.92,2.06,5.8,4.42)
# 7
s=base("06  Training & selection","Cross-validated tuning on training data only — no test-set tuning.",7)
bigpoint(s,1,"Logistic Regression","Scaled linear baseline; tune regularization strength C.",2.19)
bigpoint(s,2,"Random Forest","Nonlinear ensemble; tune depth, trees and leaf size.",3.43)
bigpoint(s,3,"Choose by training CV macro-F1",f"Selected: {selected} (3-fold stratified grid search).",4.7)
# 8
s=base("07  Held-out model evaluation","Computed metrics below follow the saved dataset mode and model artifacts.",8)
colxs=[.72,4.45,7.05,9.58]
for i,(h,w) in enumerate([("MODEL",3.6),("CV F1",2.3),("TEST ACC",2.3),("TEST F1",2.2)]):
    rectangle(s,colxs[i],2.25,w,.59,NAVY,False);tbox(s,h,colxs[i]+.12,2.36,w-.12,.4,13,WHITE,True)
for rowi,(name,val) in enumerate(metrics["results"].items()):
    y=2.9+rowi*.88
    for i,(v,w) in enumerate([(name,3.6),(f"{val['cv_f1_macro']:.3f}",2.3),(f"{val['accuracy']:.3f}",2.3),(f"{val['f1_macro']:.3f}",2.2)]):
        rectangle(s,colxs[i],y,w,.8,SOFT if rowi%2==0 else PRGB(231,242,242))
        tbox(s,v,colxs[i]+.14,y+.16,w-.16,.47,17,TEAL if i>0 else NAVY,i==0)
tbox(s,f"Selected by CV: {selected}.  Held-out test size: {metrics['test_size']} records.",.85,5.12,11.7,.84,20,NAVY,True)
if demo:tbox(s,"Numbers above = SYNTHETIC testing only; not official empirical evidence.",.85,6.09,11.8,.38,14,RED,True)
# 9
s=base("07  Confusion matrix","Rows = actual outcomes; columns = predicted outcomes.",9)
add_image(s,"04_confusion_matrix.png",1.0,2.08,7.0,4.55)
tbox(s,"Metrics include per-class precision, recall and F1-score.\n\nInspect which outcome is most frequently confused with another — overall accuracy alone can hide errors.",8.28,2.55,4.28,3,18,DARK)
# 10
s=base("08  Results interpretation","Predictive associations help explain the model, not the cause of dropout.",10)
add_image(s,"05_feature_importance.png",.72,2.06,7.0,4.55)
tbox(s,"Permutation importance shows how much predictive macro-F1 changes when a feature is shuffled.\n\nDo not infer that changing a predictor will necessarily prevent dropout.",8.15,2.39,4.4,3.6,18,DARK)
# 11
s=base("09  Working Streamlit application","A simple local interface, not a production university student record system.",11)
for i,(h,b) in enumerate([("1  OVERVIEW","Sample distribution and model statistics"),
 ("2  SINGLE SCREENING","Enter semester and tuition indicators"),
 ("3  BULK CSV","Upload multiple anonymous records; download results"),
 ("4  MODEL & EDA","Compare F1/accuracy, confusion matrix and importance")]):
    x=.72+(i%2)*6.3;y=2.2+(i//2)*2.12
    rectangle(s,x,y,5.95,1.79,SOFT,True);tbox(s,h,x+.25,y+.22,5.35,.42,17,TEAL,True)
    tbox(s,b,x+.24,y+.83,5.4,.66,17,DARK)
#12
s=base("10  Conclusion, limits & reproducibility","Useful research demonstration; not validated for real Kenyan student decisions.",12)
tbox(s,"Findings",.74,2.06,5.5,.47,22,TEAL,True)
tbox(s,"A reproducible ML pipeline and advisor dashboard demonstrate historical three-class screening. Performance is model- and dataset-specific.",.78,2.63,5.5,2,18,DARK)
tbox(s,"Safeguards",6.9,2.06,5.5,.47,22,TEAL,True)
tbox(s,"Portugal-only data • after 2nd semester • no student identities • human review only • no automated adverse decisions.",6.95,2.63,5.65,2.1,18,DARK)
rectangle(s,.74,5.22,11.9,.83,NAVY,True)
tbox(s,"Reproduce: setup_windows.bat   →   python train.py   →   streamlit run app.py",.94,5.45,11.15,.49,17,WHITE,True)
tbox(s,"Realinho et al. (2021), UCI 697, DOI: 10.24432/C5MC89  •  Student name / GitHub: add before submission",.78,6.36,12.1,.41,11,GREY)
slides_file=ROOT/"StudentShield_Presentation.pptx"
prs.save(slides_file)
print("Generated",report_file,"and",slides_file,"MODE:",MODE_LABEL)
